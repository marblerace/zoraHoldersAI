from __future__ import annotations

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_stdio_protocol_lists_all_guarded_tools() -> None:
    async def exercise() -> tuple[set[str], dict]:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server", "--transport", "stdio"],
        )
        async with (
            stdio_client(parameters) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            response = await session.list_tools()
            blocked = await session.call_tool(
                "run_sql",
                arguments={"query": "DROP TABLE holders"},
            )
            return {tool.name for tool in response.tools}, blocked.structuredContent

    names, blocked = asyncio.run(exercise())
    assert names == {
        "run_sql",
        "describe_schema",
        "data_freshness",
        "top_holders",
    }
    assert blocked["blocked"] is True
    assert "SELECT" in blocked["reason"]
