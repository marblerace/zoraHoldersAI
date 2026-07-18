from __future__ import annotations

import httpx
import pytest

from indexer.client import ExplorerError, ZoraExplorerClient

TOKEN = "0x7777777d57c1c6e472fa379b7b3b6c6ba3835073"
HOLDER_A = "0x88bb006e0ed0234a24cd94ccb06ed1f164b0ffd9"
HOLDER_B = "0x3a4ea7895a8d4a73c54050798c030ced29be35de"
TX_A = "0x" + "a" * 64
TX_B = "0x" + "b" * 64


def _client(handler, *, max_retries: int = 0, sleep=lambda _: None):
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return ZoraExplorerClient(
        "https://explorer.test/api/v2",
        client=http_client,
        max_retries=max_retries,
        backoff_seconds=0.01,
        sleep=sleep,
    )


def test_fetches_all_holder_pages_and_normalizes_addresses() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params:
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "address": {"hash": HOLDER_B.upper().replace("0X", "0x")},
                            "value": "7",
                        }
                    ],
                    "next_page_params": None,
                },
            )
        return httpx.Response(
            200,
            json={
                "items": [{"address": {"hash": HOLDER_A}, "value": "111"}],
                "next_page_params": {
                    "value": "7",
                    "address_hash": HOLDER_B,
                    "items_count": 50,
                },
            },
        )

    snapshot = _client(handler).fetch_holder_snapshot(TOKEN)

    assert snapshot.pages_fetched == 2
    assert [(holder.address, holder.balance) for holder in snapshot.holders] == [
        (HOLDER_B, 7),
        (HOLDER_A, 111),
    ]
    assert dict(requests[1].url.params) == {
        "value": "7",
        "address_hash": HOLDER_B,
        "items_count": "50",
    }


def test_retries_retryable_status_with_exponential_backoff() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503, json={"message": "try later"})
        return httpx.Response(
            200,
            json={
                "address_hash": TOKEN,
                "name": "Zora MINTs",
                "symbol": "MINT",
                "type": "ERC-1155",
                "decimals": None,
            },
        )

    metadata = _client(handler, max_retries=2, sleep=sleeps.append).fetch_token_metadata(TOKEN)

    assert metadata.name == "Zora MINTs"
    assert metadata.decimals is None
    assert attempts == 3
    assert sleeps == [0.01, 0.02]


def test_repeated_pagination_cursor_fails_closed() -> None:
    cursor = {"value": "1", "address_hash": HOLDER_A, "items_count": 50}

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "items": [{"address": {"hash": HOLDER_A}, "value": "1"}],
                "next_page_params": cursor,
            },
        )

    with pytest.raises(ExplorerError, match="repeated a pagination cursor"):
        _client(handler).fetch_holder_snapshot(TOKEN)


@pytest.mark.parametrize("value", [None, "not-a-number", "-1"])
def test_invalid_balances_fail_the_entire_snapshot(value: object) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "items": [{"address": {"hash": HOLDER_A}, "value": value}],
                "next_page_params": None,
            },
        )

    with pytest.raises(ExplorerError, match="balance"):
        _client(handler).fetch_holder_snapshot(TOKEN)


def test_missing_items_list_fails_closed() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"next_page_params": None})

    with pytest.raises(ExplorerError, match="items list"):
        _client(handler).fetch_holder_snapshot(TOKEN)


def _transfer(tx_hash: str, block: int, *, log_index: int = 0) -> dict:
    return {
        "transaction_hash": tx_hash,
        "log_index": log_index,
        "from": {"hash": HOLDER_A},
        "to": {"hash": HOLDER_B},
        "total": {"token_id": "1", "value": "3"},
        "block_number": block,
        "timestamp": "2026-07-16T08:30:00.000000Z",
        "method": "safeTransferFrom",
        "type": "token_transfer",
    }


def test_transfer_sync_rereads_watermark_block_and_stops_at_older_history() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params:
            return httpx.Response(
                200,
                json={
                    "items": [_transfer(TX_B, 100), _transfer("0x" + "c" * 64, 99)],
                    "next_page_params": {"block_number": 50, "index": 0},
                },
            )
        return httpx.Response(
            200,
            json={
                "items": [_transfer(TX_A, 105)],
                "next_page_params": {"block_number": 100, "index": 0},
            },
        )

    snapshot = _client(handler).fetch_transfer_snapshot(TOKEN, since_block_inclusive=100)

    assert snapshot.pages_fetched == 2
    assert [(event.tx_hash, event.block_number) for event in snapshot.transfers] == [
        (TX_B, 100),
        (TX_A, 105),
    ]
    assert len(requests) == 2


def test_malformed_transfer_fails_the_incremental_window() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        item = _transfer(TX_A, 105)
        item["timestamp"] = "not-a-time"
        return httpx.Response(200, json={"items": [item], "next_page_params": None})

    with pytest.raises(ExplorerError, match="timestamp"):
        _client(handler).fetch_transfer_snapshot(TOKEN)
