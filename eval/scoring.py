"""Deterministic execution-result scorers used by the eval runner."""

from __future__ import annotations

import json
import math
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any


def result_set_match(
    actual: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    expected: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    decimal_places: int = 6,
) -> bool:
    """Compare order/alias-insensitive row multisets with rounded numeric values."""

    return _row_multiset(actual, decimal_places) == _row_multiset(expected, decimal_places)


def numeric_match(
    actual: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    expected: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    relative_tolerance: Decimal = Decimal("0.0001"),
    absolute_tolerance: Decimal = Decimal("0.000001"),
) -> bool:
    """Find the expected scalar among candidate cells within a narrow tolerance."""

    expected_value = _single_numeric(expected)
    if expected_value is None:
        return False
    allowed_delta = max(absolute_tolerance, abs(expected_value) * relative_tolerance)
    return any(
        abs(actual_value - expected_value) <= allowed_delta
        for actual_value in _numeric_values(actual)
    )


def expected_answer_contains(answer: str, fragments: list[str]) -> bool:
    normalized = answer.casefold()
    return all(fragment.casefold() in normalized for fragment in fragments)


def _row_multiset(rows: Any, decimal_places: int) -> Counter[str]:
    canonical_rows: list[str] = []
    for row in rows:
        # SQL aliases and projection order can differ for semantically identical queries.
        values = sorted(
            (_canonical_value(value, decimal_places) for value in row.values()),
            key=lambda value: json.dumps(value, sort_keys=True, default=str),
        )
        canonical_rows.append(json.dumps(values, sort_keys=True, default=str))
    return Counter(canonical_rows)


def _canonical_value(value: Any, decimal_places: int) -> Any:
    if value is None or isinstance(value, bool):
        return value
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return str(value).casefold() if isinstance(value, str) else value
    if not numeric.is_finite():
        return str(numeric)
    quantum = Decimal(1).scaleb(-decimal_places)
    return str(numeric.quantize(quantum).normalize())


def _single_numeric(rows: Any) -> Decimal | None:
    if len(rows) != 1 or len(rows[0]) != 1:
        return None
    value = next(iter(rows[0].values()))
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return numeric if numeric.is_finite() else None


def _numeric_values(rows: Any) -> list[Decimal]:
    values: list[Decimal] = []
    for row in rows:
        for value in row.values():
            if value is None or isinstance(value, bool):
                continue
            try:
                numeric = Decimal(str(value))
            except (InvalidOperation, ValueError):
                continue
            if numeric.is_finite():
                values.append(numeric)
    return values


def percentile(values: list[float], percentile_value: float) -> float | None:
    """Return nearest-rank percentile without a heavy statistics dependency."""

    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile_value * len(ordered)))
    return ordered[rank - 1]
