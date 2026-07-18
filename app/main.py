"""FastAPI entry point for health and protected indexer controls."""

from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from agent.cache import cache_metrics
from agent.service import TextToSQLAgent
from app.config import Settings, get_settings
from db.core import freshness_snapshot
from indexer.scheduler import create_scheduler
from indexer.service import sync_once

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    scheduler = None
    if settings.enable_scheduler:
        scheduler = create_scheduler(settings)
        scheduler.start()
        logger.info(
            "indexer_scheduler_started interval_minutes=%s",
            settings.sync_interval_minutes,
        )
    yield
    if scheduler is not None:
        scheduler.shutdown(wait=False)


_configure_logging()
app = FastAPI(
    title="On-chain Text-to-SQL Analytics Agent",
    version="0.2.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4000)


class TokenUsageResponse(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_write_input_tokens: int
    total_tokens: int


class AskResponse(BaseModel):
    answer: str
    sql: str | None
    columns: list[str]
    rows: list[dict[str, Any]]
    token_usage: TokenUsageResponse
    cost_usd: float | None
    latency_ms: int
    retries: int
    provider: str
    model: str
    data_as_of: str | None
    status: str
    guard_rejection: str | None = None
    error: str | None = None
    reason: str | None = None
    last_error: str | None = None
    served_from_cache: bool = False
    citations: list[str] = Field(default_factory=list)


def require_admin_token(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
    x_admin_token: Annotated[str | None, Header()] = None,
) -> None:
    """Accept either a bearer token or X-Admin-Token for manual refreshes."""

    supplied = x_admin_token
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    expected = settings.admin_sync_token.get_secret_value()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin sync token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/health")
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    """Report database reachability, data freshness, and the last sync outcome."""

    try:
        snapshot = await run_in_threadpool(freshness_snapshot, settings)
    except psycopg.Error as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.selected_llm_model,
        "answer_cache": cache_metrics.snapshot(),
        **snapshot,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """Run the bounded text-to-SQL agent and return its full observable payload."""

    agent = TextToSQLAgent(settings)
    result = await run_in_threadpool(agent.ask, request.question.strip())
    return result.to_dict()


@app.post("/admin/sync", dependencies=[Depends(require_admin_token)])
async def admin_sync(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    """Trigger a synchronous refresh without blocking the event loop."""

    result = await run_in_threadpool(sync_once, settings)
    if result.status in {"failed", "partial"}:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=result.to_dict(),
        )
    return result.to_dict()
