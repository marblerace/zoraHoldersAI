BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tokens (
    token_address   TEXT PRIMARY KEY,
    chain           TEXT NOT NULL DEFAULT 'zora',
    name            TEXT,
    symbol          TEXT,
    token_type      TEXT,
    decimals        INTEGER CHECK (decimals IS NULL OR decimals >= 0),
    created_at      TIMESTAMPTZ,
    last_synced_at  TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS holders (
    token_address   TEXT NOT NULL REFERENCES tokens(token_address) ON DELETE CASCADE,
    holder_address  TEXT NOT NULL,
    balance         NUMERIC(78, 0) NOT NULL CHECK (balance >= 0),
    balance_decimal NUMERIC NOT NULL CHECK (balance_decimal >= 0),
    first_seen_at   TIMESTAMPTZ NOT NULL,
    last_updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (token_address, holder_address)
);

CREATE TABLE IF NOT EXISTS transfers (
    tx_hash         TEXT NOT NULL,
    log_index       INTEGER NOT NULL,
    token_id        TEXT NOT NULL,
    token_address   TEXT NOT NULL REFERENCES tokens(token_address) ON DELETE CASCADE,
    from_address    TEXT NOT NULL,
    to_address      TEXT NOT NULL,
    amount          NUMERIC(78, 0) NOT NULL CHECK (amount >= 0),
    block_number    BIGINT NOT NULL,
    block_time      TIMESTAMPTZ NOT NULL,
    method          TEXT,
    event_type      TEXT,
    PRIMARY KEY (tx_hash, log_index, token_id)
);

CREATE TABLE IF NOT EXISTS sync_runs (
    id              BIGSERIAL PRIMARY KEY,
    token_address   TEXT NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    status          TEXT NOT NULL CHECK (
        status IN ('running', 'succeeded', 'partial', 'failed', 'skipped')
    ),
    pages_fetched   INTEGER NOT NULL DEFAULT 0,
    rows_fetched    INTEGER NOT NULL DEFAULT 0,
    rows_upserted   INTEGER NOT NULL DEFAULT 0,
    rows_deleted    INTEGER NOT NULL DEFAULT 0,
    transfer_pages_fetched INTEGER NOT NULL DEFAULT 0,
    transfers_fetched INTEGER NOT NULL DEFAULT 0,
    transfers_upserted INTEGER NOT NULL DEFAULT 0,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS query_logs (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    question        TEXT NOT NULL,
    provider        TEXT,
    model           TEXT,
    status          TEXT NOT NULL DEFAULT 'succeeded',
    final_sql       TEXT,
    tokens_in       INTEGER,
    tokens_out      INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    cost_usd        NUMERIC(12, 8),
    latency_ms      INTEGER,
    retries         INTEGER NOT NULL DEFAULT 0,
    guard_rejection TEXT,
    error           TEXT
);

ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS reason TEXT;
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS served_from_cache BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS rows_returned INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS answer_cache (
    cache_key           TEXT PRIMARY KEY,
    normalized_question TEXT NOT NULL,
    token_address       TEXT NOT NULL,
    schema_hash         TEXT NOT NULL,
    response_json       JSONB NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ NOT NULL,
    last_accessed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    hit_count           BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS embeddings (
    doc_id          TEXT NOT NULL,
    chunk_id        TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    source_path     TEXT NOT NULL,
    heading         TEXT,
    content         TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding       vector(384) NOT NULL,
    tsv             TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', COALESCE(heading, '') || ' ' || content)
    ) STORED,
    indexed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (doc_id, chunk_id, embedding_model)
);

CREATE INDEX IF NOT EXISTS idx_holders_balance
    ON holders (token_address, balance DESC);
CREATE INDEX IF NOT EXISTS idx_holders_updated
    ON holders (token_address, last_updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_transfers_time
    ON transfers (token_address, block_time DESC);
CREATE INDEX IF NOT EXISTS idx_sync_runs_started
    ON sync_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_query_logs_created
    ON query_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_answer_cache_expiry
    ON answer_cache (expires_at);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector
    ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 20);
CREATE INDEX IF NOT EXISTS idx_embeddings_tsv
    ON embeddings USING GIN (tsv);

COMMIT;
