"""Minimal chat UI for the on-chain analytics API."""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def _api_post(question: str) -> dict[str, Any]:
    response = httpx.post(
        f"{API_BASE_URL}/ask",
        json={"question": question},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30, show_spinner=False)
def _health() -> dict[str, Any] | None:
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def _render_assistant(payload: dict[str, Any]) -> None:
    st.markdown(payload.get("answer") or "No answer returned.")
    if payload.get("status") == "degraded":
        reason = payload.get("reason") or "unknown"
        cache_note = " A cached answer was served." if payload.get("served_from_cache") else ""
        st.warning(f"Degraded response ({reason}).{cache_note}")
    citations = payload.get("citations") or []
    if citations:
        st.caption("Sources: " + ", ".join(f"[{citation}]" for citation in citations))
    sql = payload.get("sql")
    rows = payload.get("rows") or []
    with st.expander("Query details", expanded=False):
        if sql:
            st.code(sql, language="sql")
        else:
            st.caption("No SQL was executed; the agent requested clarification.")
        if payload.get("guard_rejection"):
            st.warning(f"Guard rejection: {payload['guard_rejection']}")
        if payload.get("error"):
            st.error(payload["error"])
        if payload.get("last_error") and payload.get("last_error") != payload.get("error"):
            st.error(payload["last_error"])
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)

    cost = payload.get("cost_usd")
    cost_text = "pricing unavailable" if cost is None else f"${cost:.6f}"
    freshness = payload.get("data_as_of") or "not yet synchronized"
    st.caption(
        f"Data as of {freshness} · {payload.get('latency_ms', 0) / 1000:.2f}s · "
        f"{cost_text} · {payload.get('model', 'model unavailable')}"
    )


def main() -> None:
    st.set_page_config(
        page_title="Zora On-chain Analyst",
        page_icon="◉",
        layout="centered",
    )
    st.title("Zora On-chain Analyst")
    st.caption("Ask questions about live MINT holders and indexed transfer history.")

    health = _health()
    if health is None:
        st.warning("The analytics API is not reachable yet. It may still be starting.")
    elif not health.get("token"):
        st.info("The first on-chain sync is still running; answers may be empty until it finishes.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant(message["payload"])
            else:
                st.markdown(message["content"])

    if question := st.chat_input("e.g. Who are the top 10 current holders?"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Querying guarded on-chain analytics…"):
                    payload = _api_post(question)
            except httpx.HTTPStatusError as exc:
                try:
                    detail = exc.response.json().get("detail", str(exc))
                except ValueError:
                    detail = str(exc)
                payload = {
                    "answer": f"The API could not answer this request: {detail}",
                    "status": "failed",
                    "rows": [],
                }
            except httpx.HTTPError as exc:
                payload = {
                    "answer": f"The analytics API is unavailable: {exc}",
                    "status": "failed",
                    "rows": [],
                }
            _render_assistant(payload)
        st.session_state.messages.append({"role": "assistant", "payload": payload})


if __name__ == "__main__":
    main()
