"""FastMCP transport registration kept thin over :class:`MCPTools`."""

from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from mcp_server.tools import MCPTools


def create_server(
    settings: Settings | None = None,
    *,
    tools: MCPTools | None = None,
):
    """Create the official SDK server without starting a transport."""

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as error:
        raise RuntimeError("Install the MCP extra: pip install '.[mcp]'") from error

    resolved = settings or get_settings()
    service = tools or MCPTools(resolved)
    server = FastMCP(
        "Zora Guarded Analytics",
        instructions=(
            "Read-only Zora analytics. All open-ended SQL is AST-validated, "
            "row-capped, and executed under a SELECT-only PostgreSQL role."
        ),
        host=resolved.mcp_host,
        port=resolved.mcp_port,
        stateless_http=True,
        json_response=True,
    )

    @server.tool()
    def run_sql(query: str) -> dict[str, Any]:
        """Run one guarded, read-only PostgreSQL SELECT."""

        return service.run_sql(query)

    @server.tool()
    def describe_schema() -> dict[str, Any]:
        """Describe the allowlisted analytics schema and its watermark."""

        return service.describe_schema()

    @server.tool()
    def data_freshness() -> dict[str, Any]:
        """Return token freshness and the latest synchronization run."""

        return service.data_freshness()

    @server.tool()
    def top_holders(limit: int = 10) -> dict[str, Any]:
        """Return up to 100 current holders ordered by raw balance."""

        return service.top_holders(limit)

    return server
