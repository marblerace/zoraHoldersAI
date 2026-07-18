# Analytics Methodology

## Meaning of first seen

`holders.first_seen_at` is when this indexer first observed an address in a holder snapshot. It is not guaranteed to be the wallet's first on-chain acquisition. The indexer may have started after the wallet acquired tokens, and explorer history can have coverage gaps.

## Meaning of last updated

`holders.last_updated_at` is the time the snapshot row was refreshed. It is not a transfer timestamp and does not prove when a wallet bought, received, or last moved tokens. Use `transfers.block_time` for indexed event timing.

## Snapshot versus event history

A snapshot answers present-state questions such as current balance, current holder count, and current ranking. Event history answers questions about transfers over time. Reconstructing a historical balance requires complete ordered events or historical snapshots; the current holder table alone cannot do it.

## Acquisition timing

Claims about when a wallet acquired tokens should come from indexed incoming transfer events, not `first_seen_at` or `last_updated_at`. Even the earliest indexed incoming transfer is only the earliest within available coverage and should be labeled accordingly.

## Early accumulator heuristic

An “early accumulator” is an analytical heuristic, not a stored fact. A defensible profile might combine early incoming transfer block times, multiple net inflows, retained current balance, and complete coverage. It must not infer intent, identity, or profitability, and it should cite the exact thresholds used.

## Empty and partial snapshots

An empty holder response is not automatically proof that no holders exist. The indexer rejects unexpected empty snapshots by default to protect existing data. Partial explorer runs are recorded separately, and analytics should use the most recent complete state.

## Units and rounding

Counts should remain integers. Raw token quantities preserve contract precision, while displayed token amounts use decimal scaling. Reports should name units and avoid rounding that materially changes rankings or concentration ratios.

## Query safety

All model-proposed analytics SQL passes through an abstract-syntax-tree guard. The guard permits one read-only `SELECT`, enforces an allowlist of analytics tables, blocks dangerous functions and system schemas, caps returned rows, and runs under a database role that cannot write.

## Interpretation boundaries

On-chain records establish addresses, token amounts, and events. They do not automatically establish a person's identity, coordinated control, motivation, purchase price, or legal ownership. Such claims require external evidence and should not be presented as database facts.
