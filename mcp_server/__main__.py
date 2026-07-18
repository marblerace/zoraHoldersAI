"""Console entry point for stdio and Streamable HTTP MCP transports."""

from __future__ import annotations

import argparse

from app.config import get_settings
from mcp_server.server import create_server
from observability.tracing import flush_traces


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve guarded Zora analytics over MCP")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http", "http"),
        default="stdio",
        help="stdio for desktop clients; streamable-http/http for remote clients.",
    )
    parser.add_argument("--host", help="Override MCP_HOST for HTTP transport.")
    parser.add_argument("--port", type=int, help="Override MCP_PORT for HTTP transport.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    updates = {}
    if args.host:
        updates["mcp_host"] = args.host
    if args.port:
        updates["mcp_port"] = args.port
    if updates:
        settings = settings.model_copy(update=updates)
    server = create_server(settings)
    transport = "streamable-http" if args.transport == "http" else args.transport
    try:
        server.run(transport=transport)
    except KeyboardInterrupt:
        pass
    finally:
        flush_traces()


if __name__ == "__main__":
    main()
