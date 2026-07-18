# Agent Architecture

## Text to SQL path

Quantitative questions are routed to `run_sql`. The model receives an introspected schema, proposes one PostgreSQL `SELECT`, observes structured tool rows, and summarizes only what those rows support. At most two SQL executions are allowed, including one correction after a rejected or invalid query.

## Document retrieval path

Conceptual and methodological questions are routed to `search_docs`. Dense retrieval uses a local embedding and pgvector cosine similarity. Sparse retrieval uses PostgreSQL full-text search. Reciprocal Rank Fusion combines both rankings without another hosted service.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion, or RRF, assigns each document a score based on its rank in multiple result lists. For rank `r` and constant `k`, one list contributes `1 / (k + r)`. Summing contributions rewards chunks found by both dense and sparse retrieval while avoiding incomparable raw score scales.

## Citations and provenance

Every indexed Markdown chunk stores a document ID, stable chunk ID, source path, and heading. Retrieved evidence is cited as `doc_id#chunk_id`. An answer must cite returned chunks rather than inventing a source or citing a chunk that was not retrieved.

## MCP interface

The Model Context Protocol server exposes guarded SQL, schema description, data freshness, and top-holder tools over stdio or Streamable HTTP. It is a parallel interface to the same guard and read-only executor; it does not create a privileged database path.

## Observability

Each API request and MCP tool call creates nested OpenTelemetry-style spans for model generation, guard validation, database execution, and retrieval. When Langfuse credentials are configured, its OpenTelemetry-native SDK exports the spans. With credentials absent, instrumentation is a no-op.

## Resilience and cache

Transient provider failures receive bounded exponential-backoff retries. Repeated failures open a process-wide circuit breaker. Successful answers are cached by normalized question, tracked token, and schema hash; a fresh hit skips the model, and a stale answer can be served with an explicit degraded status during provider outage.
