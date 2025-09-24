import math

import pytest

from oracle.core.normalize import normalize_to_pct


def test_normalize_to_pct_produces_100_sum():
    entries = [
        ("e4", math.log(0.6)),
        ("d4", math.log(0.3)),
        ("c4", math.log(0.1)),
    ]

    normalized = normalize_to_pct(entries)

    total = sum(value for _, value in normalized)
    assert pytest.approx(total, rel=0, abs=0.1) == 100.0


def test_normalize_preserves_order():
    entries = [
        ("e4", math.log(0.7)),
        ("d4", math.log(0.2)),
        ("c4", math.log(0.1)),
    ]

    normalized = normalize_to_pct(entries)
    assert [move for move, _ in normalized] == ["e4", "d4", "c4"]
