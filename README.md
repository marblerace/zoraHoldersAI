# On-Chain SQL + RAG Analytics Agent

Ask questions in plain English about live Zora token data or the project's analytics
methodology. The agent routes quantitative questions to guarded SQL and conceptual questions to
hybrid document retrieval, then returns the answer together with SQL rows and/or cited chunks.

The model provider is swappable: Anthropic and OpenAI API adapters are available, while
`claude_code` uses an authenticated Claude Code Pro/Max subscription without API-token billing.

> **Tests:** 100 tests currently pass, including the SQL attack suite, circuit/cache behavior,
> MCP protocol handshake, retrieval provenance, and eval scorers.

## What this demonstrates

| Capability | Implementation |
|---|---|
| Bounded agentic tool use | `run_sql` + `search_docs`, with at most two tool executions in `agent/service.py` |
| Production SQL safety | Fail-closed SQLGlot AST policy, allowlisted tables, row/timeout caps, and a SELECT-only role |
| Hybrid RAG / vector database | pgvector cosine search + PostgreSQL FTS + Reciprocal Rank Fusion in `retrieval/` |
| MCP integration | Official MCP Python SDK, FastMCP, stdio + Streamable HTTP in `mcp_server/` |
| Production observability | OpenTelemetry-native Langfuse spans, structured stdout, and `query_logs` |
| Resilience and cost control | Transient retries, provider circuit breaker, normalized answer cache, stale fallback |
| Measured quality | A 44-case adversarial SQL harness plus a real 28-case retrieval ablation in `eval/` |
| Provider independence | Anthropic, OpenAI, or local Claude Code subscription through the same `LLMClient` protocol |

## Architecture

```mermaid
flowchart LR
    Z[Zora explorer] --> I[Scheduled indexer]
    I --> P[(Postgres + pgvector)]
    U[Streamlit / API client] --> A[FastAPI /ask]
    A --> G[Bounded agent]
    G -->|run_sql| S[AST SQL guard]
    S -->|read-only transaction| P
    G -->|search_docs| R[Dense + FTS + RRF]
    R -->|guarded SELECTs| S
    P --> G
    G --> U
    C[MCP client] --> M[FastMCP stdio / HTTP]
    M --> S
    T[OpenTelemetry spans] --> L[Langfuse optional]
    G --> T
    M --> T
```

## Quickstart

Prerequisites: Docker Desktop and Python 3.11+.

```bash
cp .env.example .env
python3 -m venv .venv
.venv/bin/pip install -e '.[all,dev]'
docker compose up -d postgres
```

New database volumes initialize the schema and roles automatically. To upgrade an existing project
volume after pulling this version, apply the additive schema once:

```bash
docker compose exec -T postgres \
  psql -v ON_ERROR_STOP=1 -U postgres -d zora_analytics -f /opt/zora/schema.sql
```

Index the bundled Markdown corpus. The first command allows one download of the 67 MB quantized BGE
ONNX model; subsequent embedding runs work from the local cache without an API key:

```bash
.venv/bin/zora-retrieval-index --allow-model-download
```

### Run with a Claude subscription

Confirm that the host CLI is authenticated:

```bash
claude auth status
```

Set `LLM_PROVIDER=claude_code` in `.env`, then run the application on the host so it can invoke your
logged-in `claude` binary:

```bash
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
API_BASE_URL=http://127.0.0.1:8000 \
  .venv/bin/streamlit run ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open [the API docs](http://127.0.0.1:8000/docs) or
[the chat UI](http://127.0.0.1:8501). Claude Code subscription usage reports
`cost_usd: null`; it does not pretend subscription calls have API-token cost.

### Run with an API provider

Set either `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` and the matching `LLM_PROVIDER`. The fully
containerized path is then. Compose mounts the host `.cache/fastembed` directory read-only, so the
model cached during the indexing step is reused without another download:

```bash
docker compose up --build
```

Health and direct query examples:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Who are the top 10 current holders?"}'
```

## Guardrails

Every model- or MCP-supplied query goes through one `SQLGuard` and one `SQLExecutor`. The guard:

- accepts exactly one read-only `SELECT`;
- allowlists `tokens`, `holders`, `transfers`, and retrieval `embeddings`;
- blocks writes, DDL, system schemas, cross-database references, locks, and dangerous functions;
- preserves pgvector's `<=>` cosine operator only on `embeddings.embedding`;
- injects or clamps `LIMIT`, while the executor sets `TRANSACTION READ ONLY` and a statement timeout.

The database reader role is a separate safety boundary. A parser regression still cannot turn the
reader credential into a writer.

## Resilience and cache

Provider calls receive transient-only exponential-backoff retries. After four consecutive failures
by default, the process-wide provider/model circuit opens for 30 seconds. These values are controlled
by `LLM_PROVIDER_RETRY_ATTEMPTS`, `LLM_CIRCUIT_FAILURE_THRESHOLD`, and
`LLM_CIRCUIT_RESET_SECONDS`.

Successful responses are cached in `answer_cache` by a hash of:

1. case-folded, punctuation/whitespace-normalized question;
2. tracked token address;
3. current introspected schema hash.

A fresh hit skips the model. If the provider is unavailable, an expired last-good response can be
served explicitly as stale. Provider and guard failures return HTTP 200 with a machine-readable
contract instead of a 5xx:

```json
{
  "status": "degraded",
  "answer": "I couldn't answer that confidently.",
  "reason": "provider_unavailable",
  "last_error": "...",
  "served_from_cache": false
}
```

`GET /health` includes answer-cache lookups, hits, stale hits, and process-local hit rate.

## Observability

`observability/tracing.py` is a deliberately thin span API. One trace covers every `/ask` or MCP
call, with nested `llm.generate`, `tool.run_sql`, `tool.search_docs`, `guard.validate`, `db.execute`,
and `retrieval.*` spans. Provider/model, tokens, nullable cost, latency, retries, guard decision, and
row counts use the same values returned by the API and written to `query_logs`.

Langfuse is optional. If any of `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, or `LANGFUSE_HOST` is
missing, tracing runs as a no-op and cannot break a request. For Langfuse Cloud, create a project,
put its three values in `.env`, and restart the API.

For local Langfuse v3:

```bash
docker compose --profile observability up -d
```

Open [localhost:3000](http://localhost:3000), create the first project/key pair, place the keys in
`.env`, and set `LANGFUSE_HOST=http://localhost:3000` for a host-run API. A containerized API should
use `http://langfuse-web:3000`. Change all `LANGFUSE_*` local secrets before exposing that profile.

> **Langfuse screenshot placeholder:** add a captured `/ask` trace here before publishing the repo.

Langfuse's current Python SDK is OpenTelemetry-native, so the instrumentation is not coupled to a
custom agent framework and can coexist with another OTel backend.

## MCP server

The MCP server is a parallel interface to the same guarded executor. It exposes:

- `run_sql(query)` — structured rows or a structured guard rejection;
- `describe_schema()` — the runtime-introspected analytics schema;
- `data_freshness()` — token watermark plus latest sync run;
- `top_holders(limit=10)` — a curated query, still passed through the guard.

Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "zora-analytics": {
      "command": "/absolute/path/to/zoraHoldersAI/.venv/bin/zora-mcp",
      "args": ["--transport", "stdio"],
      "env": {
        "DATABASE_URL": "postgresql://zora_app:zora_app@localhost:55432/zora_analytics",
        "READ_ONLY_DATABASE_URL": "postgresql://zora_reader:zora_reader@localhost:55432/zora_analytics"
      }
    }
  }
}
```

Replace only the absolute repository path and, if necessary, the mapped PostgreSQL port. Any
MCP-compatible client can use the same stdio command. Streamable HTTP is one line:

```bash
.venv/bin/zora-mcp --transport streamable-http --host 127.0.0.1 --port 8001
```

Clients connect to `http://127.0.0.1:8001/mcp`.

## Hybrid retrieval

The curated corpus in `retrieval/corpus/` covers protocol scope, on-chain terminology, timestamp
semantics, safety, and system methodology. Heading-aware chunks keep stable provenance such as
`methodology#meaning-of-first-seen`.

For a query, the retriever runs two guarded `SELECT`s over the same PostgreSQL instance:

1. BGE-small dense cosine search through pgvector;
2. English full-text search through a generated `tsvector` + GIN index;
3. deterministic RRF fusion; optionally, a local FastEmbed cross-encoder reranker.

No paid embedding API is required. `EMBEDDINGS_PROVIDER=openai` remains an explicit option, while
`fastembed` is the default. With `FASTEMBED_LOCAL_FILES_ONLY=true`, a cached BGE model is used without
network; if it is not cached, a deterministic local hash embedding keeps the feature fail-soft and is
clearly labeled as a fallback rather than BGE benchmark output.

### Retrieval eval results

The table below came from a real local run on 2026-07-18 against 28 golden questions and 33 indexed
chunks. It was generated with `python -m eval.run_retrieval --allow-model-download --judge`; no
values were hand-waved.

<!-- RETRIEVAL_RESULTS_START -->
Real retrieval run using `BAAI/bge-small-en-v1.5`; ranking cutoff k=5.

| Configuration | Precision@5 | Recall@5 | MRR | Groundedness | Status |
|---|---:|---:|---:|---:|---|
| Dense only | 22.14% | 89.29% | 0.8810 | 96.43% | completed |
| Dense + FTS (RRF) | 22.86% | 92.86% | 0.9315 | 92.86% | completed |
| Hybrid + reranker | 23.57% | 94.64% | 0.9643 | 96.43% | completed |

An unavailable reranker is reported explicitly and is never relabeled as a reranked run.
<!-- RETRIEVAL_RESULTS_END -->

![Retrieval ablation](docs/assets/retrieval_ablation.png)

Run it again with:

```bash
.venv/bin/zora-retrieval-eval --allow-model-download
```

Add `--judge` to generate evidence-only answers and reuse the provider-neutral groundedness judge.
The default gate fails if hybrid MRR drops below dense-only MRR. Raw per-question results live in
`eval/results/retrieval-*.json`.

## SQL eval results

<!-- EVAL_RESULTS_START -->
The expanded benchmark contains 44 cases: 37 executable SQL questions and 7 questions that
must be clarified because the requested metric is ambiguous or absent from the schema. Its first
full run was intentionally excluded after the Claude Code subscription hit a session limit at case
27; circuit-breaker fallbacks are resilience behavior, not model-quality measurements.

Evaluation-engineering highlights from the last complete measured runs:

- Replaced string-based SQL grading with execution-result comparison against reference queries.
- Kept result-set matching strict while fixing numeric value-location and tolerance bugs.
- Separated answer faithfulness from SQL correctness and raised measured groundedness from 26.32%
  to 86.84% without weakening the correctness scorer.
- Added multi-hop joins, unavailable-data rejection, and deliberate ambiguity cases to the golden
  set instead of optimizing for a perfect headline score.
- Improved retrieval Recall@5 from 89.29% to 94.64% with hybrid retrieval plus reranking (+5.35
  percentage points), with sparse retrieval changing 26 of 28 rankings.

Run `.venv/bin/python -m eval.run` after the Claude subscription window resets to publish the full
44-case table here.
<!-- EVAL_RESULTS_END -->

SQL execution accuracy compares each generated query's actual result set with a hand-written
reference query. It is order- and alias-insensitive and tolerance-aware for numeric answers.

## Repository structure

```text
agent/           bounded tool loop, cache, circuit breaker, prompts
app/             FastAPI: /ask, /admin/sync, /health
db/              Postgres/pgvector schema, roles, schema introspection
eval/            SQL golden eval + retrieval golden set/ablation
indexer/         scheduled Zora holder/transfer synchronization
llm/             Anthropic, OpenAI, and Claude Code adapters
mcp_server/      official FastMCP server and CLI
observability/   Langfuse/Otel spans + structured/query-table logging
retrieval/       corpus, chunking, embeddings, indexing, dense/FTS/RRF
sql_guard/       fail-closed AST guard + read-only executor
ui/              Streamlit chat
tests/           package-mirrored unit and protocol tests
```

## Development

```bash
.venv/bin/pip install -e '.[all,dev]'
.venv/bin/pytest -q
.venv/bin/ruff check .
docker compose config --quiet
```

The optional dependency groups are `observability`, `mcp`, and `retrieval`; `all` is used by the
runtime Docker image. No LangChain or LlamaIndex dependency is used.
