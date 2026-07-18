"""Scheduled ingestion of live holder snapshots from the Zora explorer."""

from indexer.service import SyncResult, sync_once

__all__ = ["SyncResult", "sync_once"]
