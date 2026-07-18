"""Command-line entry points for one-shot and scheduled indexing."""

from __future__ import annotations

import argparse
import json
import logging

from apscheduler.schedulers.blocking import BlockingScheduler

from app.config import get_settings
from indexer.service import sync_once


def _run_once() -> int:
    result = sync_once(get_settings())
    print(json.dumps(result.to_dict(), default=str))
    return 0 if result.status in {"succeeded", "skipped"} else 1


def _run_scheduler() -> int:
    settings = get_settings()
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        sync_once,
        trigger="interval",
        minutes=settings.sync_interval_minutes,
        kwargs={"settings": settings},
        id="zora-holder-sync",
        coalesce=True,
        max_instances=1,
    )
    sync_once(settings)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Zora holder analytics data")
    parser.add_argument("command", choices=("sync", "schedule"), nargs="?", default="sync")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    exit_code = _run_once() if args.command == "sync" else _run_scheduler()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
