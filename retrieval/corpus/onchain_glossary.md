# On-chain Analytics Glossary

## Holder

A holder is an address with a positive current token balance in the indexed snapshot. One human or organization may control several addresses, and a smart contract can also be a holder, so holder count is not the same as unique people.

## Raw and decimal balances

Smart contracts store token quantities as integers. The raw `balance` preserves that exact integer. `balance_decimal` divides the raw value by ten raised to the token's decimals and is the appropriate field for human-readable amounts and averages.

## Transfer

A transfer is an on-chain event moving a token amount from one address to another. It establishes an event at a block and timestamp, but it does not by itself establish the economic reason, beneficial owner, or whether two addresses share control.

## Mint

A mint creates or issues token units. In standard transfer-event semantics, a transfer whose sender is the zero address is treated as a mint. The recipient receives newly represented supply rather than units sent by an ordinary wallet.

## Burn

A burn removes token units from circulating ownership. In standard transfer-event semantics, a transfer whose recipient is the zero address is treated as a burn. Sending to an inaccessible address that is not the canonical zero address may require separate interpretation.

## Zero address

The zero address is `0x0000000000000000000000000000000000000000`. It is a protocol sentinel rather than a normal wallet. Its position as sender or recipient is commonly used to distinguish mint and burn events.

## Data watermark

A data watermark is the latest source time through which a dataset is known to be synchronized. In this project, `tokens.last_synced_at` communicates when the tracked token snapshot was last refreshed; it is not the timestamp of every underlying transfer.

## Token decimals

Token decimals specify the display scaling between an integer contract value and a human-readable amount. With 18 decimals, a raw value of 1,000,000,000,000,000,000 represents one token. Decimal scaling changes presentation, not ownership.

## Block time

Block time is the timestamp associated with the block containing an event. It is the correct field for transfer time-window queries. A block timestamp can differ from when an indexer observed or stored the event.

## Wallet and contract address

An externally owned account is controlled by a private key, while a contract address executes deployed code. Both can appear in holder and transfer data. Address type alone does not identify the real-world controller.

## Holder concentration

Holder concentration describes how much supply is controlled by the largest indexed addresses. A top-holder share or concentration ratio should use current balances and a clearly defined denominator, and should acknowledge treasury, bridge, pool, burn, or contract addresses when known.
