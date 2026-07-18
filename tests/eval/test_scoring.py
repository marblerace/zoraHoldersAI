from decimal import Decimal

from eval.scoring import numeric_match, percentile, result_set_match


def test_result_set_match_ignores_row_order_aliases_and_small_rounding() -> None:
    actual = [
        {"wallet": "0xabc", "amount": "1.23456749"},
        {"wallet": "0xdef", "amount": Decimal("2")},
    ]
    expected = [
        {"balance": "2.0000001", "address": "0xDEF"},
        {"balance": "1.2345674", "address": "0xABC"},
    ]

    assert result_set_match(actual, expected) is True


def test_result_set_match_preserves_duplicate_multiplicity() -> None:
    assert result_set_match([{"x": 1}, {"x": 1}], [{"x": 1}]) is False


def test_result_set_match_rejects_extra_projected_values() -> None:
    assert result_set_match([{"answer": 1, "extra": 2}], [{"answer": 1}]) is False


def test_numeric_match_uses_relative_and_absolute_tolerance() -> None:
    assert numeric_match([{"actual": "100.005"}], [{"expected": "100"}]) is True
    assert numeric_match([{"actual": "100.1"}], [{"expected": "100"}]) is False
    assert numeric_match([{"actual": "0.0000005"}], [{"expected": "0"}]) is True


def test_numeric_match_finds_expected_value_among_candidate_columns() -> None:
    assert numeric_match([{"address": "0xabc", "balance": 111}], [{"value": 111}]) is True
    assert numeric_match([{"count": 1774, "total": 6523}], [{"value": 6523}]) is True
    assert numeric_match([{"count": 1774, "total": 6523}], [{"value": 99}]) is False


def test_numeric_match_requires_one_expected_scalar() -> None:
    assert numeric_match([{"x": 1}], [{"x": 1, "y": 2}]) is False


def test_nearest_rank_percentile() -> None:
    assert percentile([1, 2, 3, 4, 100], 0.95) == 100
    assert percentile([], 0.95) is None
