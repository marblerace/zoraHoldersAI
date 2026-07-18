"""Request cost estimates derived from provider-reported token usage."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.config import Settings
from llm.types import TokenUsage

MILLION = Decimal(1_000_000)


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_per_million: Decimal
    output_per_million: Decimal
    cache_read_per_million: Decimal
    cache_write_per_million: Decimal


# Last verified 2026-07-16. Provider pricing changes; update this table from the
# official pricing pages before publishing new benchmark numbers.
MODEL_PRICES: dict[str, ModelPrice] = {
    # Introductory Sonnet 5 pricing is scheduled through 2026-08-31.
    "claude-sonnet-5": ModelPrice(
        input_per_million=Decimal("2.00"),
        output_per_million=Decimal("10.00"),
        cache_read_per_million=Decimal("0.20"),
        cache_write_per_million=Decimal("2.50"),
    ),
    "gpt-5.6-terra": ModelPrice(
        input_per_million=Decimal("2.50"),
        output_per_million=Decimal("15.00"),
        cache_read_per_million=Decimal("0.25"),
        cache_write_per_million=Decimal("3.125"),
    ),
}


def estimate_cost_usd(
    model: str,
    usage: TokenUsage,
    settings: Settings,
) -> Decimal | None:
    """Estimate USD cost, returning None when a custom model has no known price."""

    price = MODEL_PRICES.get(model)
    if settings.llm_input_cost_per_million is not None:
        output_price = settings.llm_output_cost_per_million
        if output_price is None:
            return None
        input_price = settings.llm_input_cost_per_million
        price = ModelPrice(
            input_per_million=input_price,
            output_per_million=output_price,
            cache_read_per_million=input_price,
            cache_write_per_million=input_price,
        )
    if price is None:
        return None

    cost = (
        Decimal(usage.input_tokens) * price.input_per_million
        + Decimal(usage.output_tokens) * price.output_per_million
        + Decimal(usage.cache_read_input_tokens) * price.cache_read_per_million
        + Decimal(usage.cache_write_input_tokens) * price.cache_write_per_million
    ) / MILLION
    return cost.quantize(Decimal("0.00000001"))
