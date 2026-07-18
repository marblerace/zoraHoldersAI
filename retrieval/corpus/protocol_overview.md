# Zora Analytics Dataset

## Project scope

This project tracks one configured token on the Zora network. It turns explorer data into a small relational analytics dataset and answers questions only from synchronized records. It is not a general blockchain indexer and it does not claim coverage for assets outside the configured token address.

## Token metadata

The `tokens` table records the tracked token address, chain, name, symbol, token type, decimal precision, creation time when available, and the latest synchronization watermark. Token decimals determine how raw integer units are converted into display-scale balances.

## Current holder snapshot

The `holders` table represents current state at the most recent successful snapshot. Each row is one token and wallet pair with a raw integer balance and a decimal-adjusted balance. A holder count from this table means wallets with current indexed balances, not every wallet that has ever owned the token.

## Transfer history

The `transfers` table stores indexed on-chain transfer events identified by transaction hash, log index, and token ID. It includes sender, recipient, raw amount, block number, block timestamp, method, and event type. Transfer history can support time-window and flow questions when synchronization coverage is complete.

## Synchronization and freshness

The indexer fetches paginated explorer data, upserts deterministic keys, and records each run in `sync_runs`. `tokens.last_synced_at` is the primary data watermark. A successful recent run improves freshness, while a partial or failed run means users should qualify conclusions with the last complete watermark.

## Dataset limitations

Explorer APIs can lag, paginate inconsistently, or omit data temporarily. Current balances do not reveal a wallet owner's identity or intent. The dataset should not be used to infer off-chain ownership, investment advice, or causal behavior without additional evidence.
