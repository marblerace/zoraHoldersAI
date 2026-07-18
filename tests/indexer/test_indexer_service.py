from __future__ import annotations

from datetime import UTC, datetime

from app.config import Settings
from indexer.client import HolderBalance, HolderSnapshot, TokenMetadata
from indexer.service import sync_once

TOKEN = "0x7777777d57c1c6e472fa379b7b3b6c6ba3835073"
HOLDER = "0x88bb006e0ed0234a24cd94ccb06ed1f164b0ffd9"


class FakeCursor:
    def __init__(self) -> None:
        self.next_row = None
        self.executions: list[tuple[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, query: str, params=None) -> None:
        self.executions.append((query, params))
        if "INSERT INTO sync_runs" in query:
            self.next_row = {"id": 7}
        elif "pg_try_advisory_lock" in query:
            self.next_row = {"acquired": True}
        else:
            self.next_row = None

    def fetchone(self):
        return self.next_row


class FakeConnection:
    def __init__(self) -> None:
        self.fake_cursor = FakeCursor()
        self.executions: list[tuple[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return self.fake_cursor

    def execute(self, query: str, params=None):
        self.executions.append((query, params))
        return self.fake_cursor


class FakeExplorer:
    def fetch_token_metadata(self, _: str) -> TokenMetadata:
        return TokenMetadata(TOKEN, "Zora MINTs", "MINT", "ERC-1155", None)

    def fetch_holder_snapshot(self, _: str) -> HolderSnapshot:
        return HolderSnapshot((HolderBalance(HOLDER, 111),), pages_fetched=2)


def test_transfer_failure_after_holder_commit_is_reported_as_partial(monkeypatch) -> None:
    connection = FakeConnection()
    synced_at = datetime(2026, 7, 16, 8, 30, tzinfo=UTC)
    monkeypatch.setattr("indexer.service.connect_writer", lambda _: connection)
    monkeypatch.setattr(
        "indexer.service._persist_holders",
        lambda *args, **kwargs: (synced_at, 1, 0),
    )

    def fail_transfers(*args, **kwargs):
        raise RuntimeError("transfer source unavailable")

    monkeypatch.setattr("indexer.service._fetch_transfers", fail_transfers)
    settings = Settings(_env_file=None, enable_scheduler=False)

    result = sync_once(settings, explorer=FakeExplorer())

    assert result.status == "partial"
    assert result.rows_fetched == 1
    assert result.rows_upserted == 1
    assert "transfer source unavailable" in (result.error or "")
    failure_updates = [
        params for query, params in connection.executions if "SET status = %s" in query
    ]
    assert failure_updates[0][0] == "partial"
