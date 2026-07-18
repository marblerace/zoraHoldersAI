"""Transactional holder snapshot synchronization."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from app.config import Settings, get_settings
from db.core import connect_writer
from indexer.client import (
    HolderSnapshot,
    TokenMetadata,
    TransferSnapshot,
    ZoraExplorerClient,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SyncResult:
    """Serializable outcome of one attempted indexer run."""

    run_id: int
    status: str
    token_address: str
    started_at: datetime
    finished_at: datetime
    pages_fetched: int = 0
    rows_fetched: int = 0
    rows_upserted: int = 0
    rows_deleted: int = 0
    transfer_pages_fetched: int = 0
    transfers_fetched: int = 0
    transfers_upserted: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sync_once(
    settings: Settings | None = None,
    *,
    explorer: ZoraExplorerClient | None = None,
) -> SyncResult:
    """Fetch a complete holder snapshot and atomically reconcile it into Postgres.

    A session-level advisory lock prevents overlapping scheduler/API runs across
    processes. Holder data changes only after every holder page is fetched and
    validated. That snapshot commits before the slower transfer backfill, so a
    transfer-source failure is reported as partial without hiding fresh holder data.
    """

    resolved = settings or get_settings()
    started_at = datetime.now(UTC)
    owned_explorer = explorer is None
    client = explorer or ZoraExplorerClient(
        resolved.zora_explorer_base_url,
        timeout_seconds=resolved.explorer_timeout_seconds,
        max_retries=resolved.explorer_max_retries,
        backoff_seconds=resolved.explorer_backoff_seconds,
        max_pages=resolved.explorer_max_pages,
        max_transfer_pages=resolved.explorer_max_transfer_pages,
    )
    holder_synced_at: datetime | None = None
    rows_fetched = 0
    rows_upserted = 0
    rows_deleted = 0
    pages_fetched = 0

    with connect_writer(resolved) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO sync_runs (token_address, started_at, status)
            VALUES (%s, %s, 'running')
            RETURNING id
            """,
            (resolved.tracked_token_address, started_at),
        )
        run_id = int(cursor.fetchone()["id"])
        lock_name = f"holder-sync:{resolved.tracked_chain}:{resolved.tracked_token_address}"
        cursor.execute(
            "SELECT pg_try_advisory_lock(hashtext(%s)::bigint) AS acquired",
            (lock_name,),
        )
        lock_acquired = bool(cursor.fetchone()["acquired"])

        if not lock_acquired:
            result = _finish_skipped(connection, run_id, resolved, started_at)
            _log_result(result)
            if owned_explorer:
                client.close()
            return result

        try:
            metadata = client.fetch_token_metadata(resolved.tracked_token_address)
            snapshot = client.fetch_holder_snapshot(resolved.tracked_token_address)
            _reject_suspicious_empty_snapshot(connection, resolved, snapshot)
            holder_synced_at, rows_upserted, rows_deleted = _persist_holders(
                connection,
                run_id=run_id,
                settings=resolved,
                metadata=metadata,
                snapshot=snapshot,
            )
            pages_fetched = snapshot.pages_fetched
            rows_fetched = len(snapshot.holders)
            transfer_snapshot = _fetch_transfers(connection, resolved, client)
            transfers_upserted, finished_at = _persist_transfers(
                connection,
                run_id=run_id,
                settings=resolved,
                snapshot=transfer_snapshot,
            )
            result = SyncResult(
                run_id=run_id,
                status="succeeded",
                token_address=resolved.tracked_token_address,
                started_at=started_at,
                finished_at=finished_at,
                pages_fetched=pages_fetched,
                rows_fetched=rows_fetched,
                rows_upserted=rows_upserted,
                rows_deleted=rows_deleted,
                transfer_pages_fetched=transfer_snapshot.pages_fetched,
                transfers_fetched=len(transfer_snapshot.transfers),
                transfers_upserted=transfers_upserted,
            )
        except Exception as exc:  # The run must be recorded even for unexpected failures.
            finished_at = datetime.now(UTC)
            error = f"{type(exc).__name__}: {exc}"[:4000]
            failure_status = "partial" if holder_synced_at else "failed"
            connection.execute(
                """
                UPDATE sync_runs
                SET status = %s, finished_at = %s, error = %s
                WHERE id = %s
                """,
                (failure_status, finished_at, error, run_id),
            )
            result = SyncResult(
                run_id=run_id,
                status=failure_status,
                token_address=resolved.tracked_token_address,
                started_at=started_at,
                finished_at=finished_at,
                pages_fetched=pages_fetched,
                rows_fetched=rows_fetched,
                rows_upserted=rows_upserted,
                rows_deleted=rows_deleted,
                error=error,
            )
            logger.exception("holder_sync_failed", extra={"sync_run_id": run_id})
        finally:
            cursor.execute(
                "SELECT pg_advisory_unlock(hashtext(%s)::bigint)",
                (lock_name,),
            )
            if owned_explorer:
                client.close()

    _log_result(result)
    return result


def _persist_holders(
    connection: Any,
    *,
    run_id: int,
    settings: Settings,
    metadata: TokenMetadata,
    snapshot: HolderSnapshot,
) -> tuple[datetime, int, int]:
    synced_at = datetime.now(UTC)
    decimals = metadata.decimals or 0
    scale = Decimal(10) ** decimals

    with connection.transaction(), connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO tokens (
                token_address, chain, name, symbol, token_type, decimals, last_synced_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (token_address) DO UPDATE SET
                chain = EXCLUDED.chain,
                name = COALESCE(EXCLUDED.name, tokens.name),
                symbol = COALESCE(EXCLUDED.symbol, tokens.symbol),
                token_type = COALESCE(EXCLUDED.token_type, tokens.token_type),
                decimals = COALESCE(EXCLUDED.decimals, tokens.decimals),
                last_synced_at = EXCLUDED.last_synced_at
            """,
            (
                metadata.address,
                settings.tracked_chain,
                metadata.name,
                metadata.symbol,
                metadata.token_type,
                metadata.decimals,
                synced_at,
            ),
        )
        cursor.execute(
            """
            CREATE TEMP TABLE current_holder_snapshot (
                holder_address TEXT PRIMARY KEY,
                balance NUMERIC(78, 0) NOT NULL,
                balance_decimal NUMERIC NOT NULL
            ) ON COMMIT DROP
            """
        )
        with cursor.copy(
            """
            COPY current_holder_snapshot (holder_address, balance, balance_decimal)
            FROM STDIN
            """
        ) as copy:
            for holder in snapshot.holders:
                copy.write_row((holder.address, holder.balance, Decimal(holder.balance) / scale))

        cursor.execute(
            """
            INSERT INTO holders (
                token_address, holder_address, balance, balance_decimal,
                first_seen_at, last_updated_at
            )
            SELECT %s, holder_address, balance, balance_decimal, %s, %s
            FROM current_holder_snapshot
            ON CONFLICT (token_address, holder_address) DO UPDATE SET
                balance = EXCLUDED.balance,
                balance_decimal = EXCLUDED.balance_decimal,
                last_updated_at = EXCLUDED.last_updated_at
            """,
            (settings.tracked_token_address, synced_at, synced_at),
        )
        rows_upserted = cursor.rowcount
        cursor.execute(
            """
            DELETE FROM holders AS stored
            WHERE stored.token_address = %s
              AND NOT EXISTS (
                  SELECT 1
                  FROM current_holder_snapshot AS fresh
                  WHERE fresh.holder_address = stored.holder_address
              )
            """,
            (settings.tracked_token_address,),
        )
        rows_deleted = cursor.rowcount
        cursor.execute(
            """
            UPDATE sync_runs
            SET pages_fetched = %s, rows_fetched = %s,
                rows_upserted = %s, rows_deleted = %s
            WHERE id = %s
            """,
            (
                snapshot.pages_fetched,
                len(snapshot.holders),
                rows_upserted,
                rows_deleted,
                run_id,
            ),
        )

    return synced_at, rows_upserted, rows_deleted


def _fetch_transfers(
    connection: Any,
    settings: Settings,
    client: ZoraExplorerClient,
) -> TransferSnapshot:
    if not settings.sync_transfers:
        return TransferSnapshot(transfers=(), pages_fetched=0)
    row = connection.execute(
        "SELECT MAX(block_number) AS block_number FROM transfers WHERE token_address = %s",
        (settings.tracked_token_address,),
    ).fetchone()
    watermark = int(row["block_number"]) if row and row["block_number"] is not None else None
    return client.fetch_transfer_snapshot(
        settings.tracked_token_address,
        since_block_inclusive=watermark,
    )


def _upsert_transfers(
    cursor: Any,
    *,
    settings: Settings,
    snapshot: TransferSnapshot,
) -> int:
    if not settings.sync_transfers:
        return 0
    cursor.execute(
        """
        CREATE TEMP TABLE current_transfer_snapshot (
            tx_hash TEXT NOT NULL,
            log_index INTEGER NOT NULL,
            token_id TEXT NOT NULL,
            from_address TEXT NOT NULL,
            to_address TEXT NOT NULL,
            amount NUMERIC(78, 0) NOT NULL,
            block_number BIGINT NOT NULL,
            block_time TIMESTAMPTZ NOT NULL,
            method TEXT,
            event_type TEXT,
            PRIMARY KEY (tx_hash, log_index, token_id)
        ) ON COMMIT DROP
        """
    )
    with cursor.copy(
        """
        COPY current_transfer_snapshot (
            tx_hash, log_index, token_id, from_address, to_address, amount,
            block_number, block_time, method, event_type
        ) FROM STDIN
        """
    ) as copy:
        for transfer in snapshot.transfers:
            copy.write_row(
                (
                    transfer.tx_hash,
                    transfer.log_index,
                    transfer.token_id,
                    transfer.from_address,
                    transfer.to_address,
                    transfer.amount,
                    transfer.block_number,
                    transfer.block_time,
                    transfer.method,
                    transfer.event_type,
                )
            )
    cursor.execute(
        """
        INSERT INTO transfers (
            tx_hash, log_index, token_id, token_address, from_address, to_address,
            amount, block_number, block_time, method, event_type
        )
        SELECT tx_hash, log_index, token_id, %s, from_address, to_address,
               amount, block_number, block_time, method, event_type
        FROM current_transfer_snapshot
        ON CONFLICT (tx_hash, log_index, token_id) DO UPDATE SET
            token_address = EXCLUDED.token_address,
            from_address = EXCLUDED.from_address,
            to_address = EXCLUDED.to_address,
            amount = EXCLUDED.amount,
            block_number = EXCLUDED.block_number,
            block_time = EXCLUDED.block_time,
            method = EXCLUDED.method,
            event_type = EXCLUDED.event_type
        """,
        (settings.tracked_token_address,),
    )
    return cursor.rowcount


def _persist_transfers(
    connection: Any,
    *,
    run_id: int,
    settings: Settings,
    snapshot: TransferSnapshot,
) -> tuple[int, datetime]:
    finished_at = datetime.now(UTC)
    with connection.transaction(), connection.cursor() as cursor:
        transfers_upserted = _upsert_transfers(
            cursor,
            settings=settings,
            snapshot=snapshot,
        )
        cursor.execute(
            """
            UPDATE sync_runs
            SET status = 'succeeded', finished_at = %s,
                transfer_pages_fetched = %s, transfers_fetched = %s,
                transfers_upserted = %s, error = NULL
            WHERE id = %s
            """,
            (
                finished_at,
                snapshot.pages_fetched,
                len(snapshot.transfers),
                transfers_upserted,
                run_id,
            ),
        )
    return transfers_upserted, finished_at


def _reject_suspicious_empty_snapshot(
    connection: Any,
    settings: Settings,
    snapshot: HolderSnapshot,
) -> None:
    if snapshot.holders or settings.allow_empty_holder_snapshot:
        return
    row = connection.execute(
        "SELECT COUNT(*) AS count FROM holders WHERE token_address = %s",
        (settings.tracked_token_address,),
    ).fetchone()
    if row and int(row["count"]) > 0:
        raise RuntimeError(
            "Explorer returned an empty snapshot for a token with stored holders; "
            "refusing to delete the last good snapshot"
        )


def _finish_skipped(
    connection: Any,
    run_id: int,
    settings: Settings,
    started_at: datetime,
) -> SyncResult:
    finished_at = datetime.now(UTC)
    reason = "Another sync already holds the database advisory lock"
    connection.execute(
        """
        UPDATE sync_runs
        SET status = 'skipped', finished_at = %s, error = %s
        WHERE id = %s
        """,
        (finished_at, reason, run_id),
    )
    return SyncResult(
        run_id=run_id,
        status="skipped",
        token_address=settings.tracked_token_address,
        started_at=started_at,
        finished_at=finished_at,
        error=reason,
    )


def _log_result(result: SyncResult) -> None:
    logger.info(json.dumps({"event": "holder_sync", **result.to_dict()}, default=str))
