"""Resilient client for the Blockscout-compatible Zora explorer API."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

ADDRESS_PATTERN = re.compile(r"^0x[a-fA-F0-9]{40}$")
TRANSACTION_HASH_PATTERN = re.compile(r"^0x[a-fA-F0-9]{64}$")
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class ExplorerError(RuntimeError):
    """Raised when a complete, trustworthy explorer snapshot cannot be fetched."""


@dataclass(frozen=True, slots=True)
class TokenMetadata:
    """Token fields exposed by the explorer token endpoint."""

    address: str
    name: str | None
    symbol: str | None
    token_type: str | None
    decimals: int | None


@dataclass(frozen=True, slots=True)
class HolderBalance:
    """One holder and its raw integer balance."""

    address: str
    balance: int


@dataclass(frozen=True, slots=True)
class HolderSnapshot:
    """A fully paginated holder snapshot."""

    holders: tuple[HolderBalance, ...]
    pages_fetched: int


@dataclass(frozen=True, slots=True)
class TransferEvent:
    """One immutable ERC-1155 transfer event from the explorer."""

    tx_hash: str
    log_index: int
    token_id: str
    from_address: str
    to_address: str
    amount: int
    block_number: int
    block_time: datetime
    method: str | None
    event_type: str | None


@dataclass(frozen=True, slots=True)
class TransferSnapshot:
    """A complete history or incremental transfer page window."""

    transfers: tuple[TransferEvent, ...]
    pages_fetched: int


class ZoraExplorerClient:
    """Fetch token metadata and all holder pages with bounded retries."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 20.0,
        max_retries: int = 4,
        backoff_seconds: float = 0.5,
        max_pages: int = 10_000,
        max_transfer_pages: int | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._max_pages = max_pages
        self._max_transfer_pages = max_transfer_pages or max_pages
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            headers={
                "accept": "application/json",
                "user-agent": "onchain-text2sql-indexer/0.1",
            },
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ZoraExplorerClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_token_metadata(self, token_address: str) -> TokenMetadata:
        """Fetch and validate the tracked token's descriptive metadata."""

        normalized_address = self._validate_address(token_address, field="token address")
        payload = self._request_json(f"tokens/{normalized_address}")
        response_address = payload.get("address_hash") or normalized_address
        response_address = self._validate_address(response_address, field="metadata address")
        if response_address != normalized_address:
            raise ExplorerError("Explorer returned metadata for a different token address")

        raw_decimals = payload.get("decimals")
        try:
            decimals = None if raw_decimals in (None, "") else int(raw_decimals)
        except (TypeError, ValueError) as exc:
            raise ExplorerError("Explorer returned invalid token decimals") from exc
        if decimals is not None and decimals < 0:
            raise ExplorerError("Explorer returned negative token decimals")

        return TokenMetadata(
            address=normalized_address,
            name=self._optional_string(payload.get("name")),
            symbol=self._optional_string(payload.get("symbol")),
            token_type=self._optional_string(payload.get("type")),
            decimals=decimals,
        )

    def fetch_holder_snapshot(self, token_address: str) -> HolderSnapshot:
        """Fetch every holder page, failing closed if pagination is malformed."""

        normalized_address = self._validate_address(token_address, field="token address")
        params: dict[str, Any] | None = None
        seen_cursors: set[str] = set()
        holders: dict[str, HolderBalance] = {}
        pages_fetched = 0

        while True:
            if pages_fetched >= self._max_pages:
                raise ExplorerError(f"Holder pagination exceeded {self._max_pages} pages")

            payload = self._request_json(
                f"tokens/{normalized_address}/holders",
                params=params,
            )
            pages_fetched += 1
            items = payload.get("items")
            if not isinstance(items, list):
                raise ExplorerError("Holder response is missing an items list")

            for item in items:
                holder = self._parse_holder(item)
                holders[holder.address] = holder

            next_page = payload.get("next_page_params")
            if next_page is None:
                break
            if not isinstance(next_page, dict) or not next_page:
                raise ExplorerError("Explorer returned malformed next_page_params")

            cursor_key = json.dumps(next_page, sort_keys=True, separators=(",", ":"))
            if cursor_key in seen_cursors:
                raise ExplorerError("Explorer repeated a pagination cursor")
            seen_cursors.add(cursor_key)
            params = next_page

        return HolderSnapshot(
            holders=tuple(sorted(holders.values(), key=lambda holder: holder.address)),
            pages_fetched=pages_fetched,
        )

    def fetch_transfer_snapshot(
        self,
        token_address: str,
        *,
        since_block_inclusive: int | None = None,
    ) -> TransferSnapshot:
        """Fetch full history or events at/after a stored block watermark.

        The API sorts newest-first. Re-reading the entire watermark block makes
        incremental sync robust to multiple logs in the latest observed block;
        primary-key upserts remove duplicates.
        """

        normalized_address = self._validate_address(token_address, field="token address")
        params: dict[str, Any] | None = None
        seen_cursors: set[str] = set()
        transfers: dict[tuple[str, int, str], TransferEvent] = {}
        pages_fetched = 0
        reached_older_block = False

        while True:
            if pages_fetched >= self._max_transfer_pages:
                raise ExplorerError(
                    f"Transfer pagination exceeded {self._max_transfer_pages} pages"
                )
            payload = self._request_json(
                f"tokens/{normalized_address}/transfers",
                params=params,
            )
            pages_fetched += 1
            items = payload.get("items")
            if not isinstance(items, list):
                raise ExplorerError("Transfer response is missing an items list")

            for item in items:
                transfer = self._parse_transfer(item)
                if (
                    since_block_inclusive is not None
                    and transfer.block_number < since_block_inclusive
                ):
                    reached_older_block = True
                    continue
                key = (transfer.tx_hash, transfer.log_index, transfer.token_id)
                transfers[key] = transfer

            next_page = payload.get("next_page_params")
            if reached_older_block or next_page is None:
                break
            if not isinstance(next_page, dict) or not next_page:
                raise ExplorerError("Explorer returned malformed transfer next_page_params")
            cursor_key = json.dumps(next_page, sort_keys=True, separators=(",", ":"))
            if cursor_key in seen_cursors:
                raise ExplorerError("Explorer repeated a transfer pagination cursor")
            seen_cursors.add(cursor_key)
            params = next_page

        return TransferSnapshot(
            transfers=tuple(
                sorted(
                    transfers.values(),
                    key=lambda event: (event.block_number, event.log_index, event.tx_hash),
                )
            ),
            pages_fetched=pages_fetched,
        )

    def _request_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}/{path.lstrip('/')}"
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.get(url, params=params)
                if response.status_code in RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                    self._backoff(response, attempt)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ExplorerError("Explorer response must be a JSON object")
                return payload
            except httpx.RequestError as exc:
                last_error = exc
                if attempt < self._max_retries:
                    self._sleep(self._delay_for(attempt))
                    continue
                break
            except httpx.HTTPStatusError as exc:
                last_error = exc
                break
            except ValueError as exc:
                raise ExplorerError("Explorer returned invalid JSON") from exc

        detail = str(last_error) if last_error else "retry budget exhausted"
        raise ExplorerError(f"Explorer request failed for {url}: {detail}") from last_error

    def _backoff(self, response: httpx.Response, attempt: int) -> None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                delay = min(float(retry_after), 60.0)
            except ValueError:
                delay = self._delay_for(attempt)
        else:
            delay = self._delay_for(attempt)
        self._sleep(delay)

    def _delay_for(self, attempt: int) -> float:
        return self._backoff_seconds * (2**attempt)

    @classmethod
    def _parse_holder(cls, item: object) -> HolderBalance:
        if not isinstance(item, dict):
            raise ExplorerError("Holder item must be a JSON object")
        address_payload = item.get("address")
        if not isinstance(address_payload, dict):
            raise ExplorerError("Holder item is missing its address object")
        address = cls._validate_address(address_payload.get("hash"), field="holder address")
        try:
            balance = int(item.get("value"))
        except (TypeError, ValueError) as exc:
            raise ExplorerError("Holder item contains a non-integer balance") from exc
        if balance < 0:
            raise ExplorerError("Holder item contains a negative balance")
        return HolderBalance(address=address, balance=balance)

    @classmethod
    def _parse_transfer(cls, item: object) -> TransferEvent:
        if not isinstance(item, dict):
            raise ExplorerError("Transfer item must be a JSON object")
        tx_hash = item.get("transaction_hash")
        if not isinstance(tx_hash, str) or not TRANSACTION_HASH_PATTERN.fullmatch(tx_hash):
            raise ExplorerError("Transfer item contains an invalid transaction hash")

        from_payload = item.get("from")
        to_payload = item.get("to")
        if not isinstance(from_payload, dict) or not isinstance(to_payload, dict):
            raise ExplorerError("Transfer item is missing from/to address objects")
        from_address = cls._validate_address(from_payload.get("hash"), field="from address")
        to_address = cls._validate_address(to_payload.get("hash"), field="to address")

        total = item.get("total")
        if not isinstance(total, dict):
            raise ExplorerError("Transfer item is missing total")
        token_id = total.get("token_id")
        if not isinstance(token_id, str) or not token_id.isdigit():
            raise ExplorerError("Transfer item contains an invalid token_id")
        try:
            amount = int(total.get("value"))
            log_index = int(item.get("log_index"))
            block_number = int(item.get("block_number"))
        except (TypeError, ValueError) as exc:
            raise ExplorerError("Transfer item contains invalid numeric fields") from exc
        if min(amount, log_index, block_number) < 0:
            raise ExplorerError("Transfer item contains negative numeric fields")

        timestamp = item.get("timestamp")
        if not isinstance(timestamp, str):
            raise ExplorerError("Transfer item contains an invalid timestamp")
        try:
            block_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ExplorerError("Transfer item contains an invalid timestamp") from exc
        if block_time.tzinfo is None:
            raise ExplorerError("Transfer timestamp must include a timezone")

        return TransferEvent(
            tx_hash=tx_hash.lower(),
            log_index=log_index,
            token_id=token_id,
            from_address=from_address,
            to_address=to_address,
            amount=amount,
            block_number=block_number,
            block_time=block_time.astimezone(UTC),
            method=cls._optional_string(item.get("method")),
            event_type=cls._optional_string(item.get("type")),
        )

    @staticmethod
    def _validate_address(value: object, *, field: str) -> str:
        if not isinstance(value, str) or not ADDRESS_PATTERN.fullmatch(value):
            raise ExplorerError(f"Invalid {field}")
        return value.lower()

    @staticmethod
    def _optional_string(value: object) -> str | None:
        return value if isinstance(value, str) and value else None
