"""APScheduler integration for periodic holder refreshes."""

from __future__ import annotations

from datetime import UTC, datetime

from apscheduler.schedulers.background import BackgroundScheduler

from app.config import Settings
from indexer.service import sync_once


def create_scheduler(settings: Settings) -> BackgroundScheduler:
    """Create a configured scheduler; the caller controls its lifecycle."""

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        sync_once,
        trigger="interval",
        minutes=settings.sync_interval_minutes,
        kwargs={"settings": settings},
        id="zora-holder-sync",
        name="Refresh Zora holder snapshot",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        next_run_time=datetime.now(UTC) if settings.sync_on_startup else None,
    )
    return scheduler
