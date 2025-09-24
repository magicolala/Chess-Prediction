import math

from oracle.core.expected import aggregate_expected_score, expected_score_cp


def test_expected_score_cp_logistic():
    zero = expected_score_cp(0)
    positive = expected_score_cp(100)
    negative = expected_score_cp(-100)

    assert math.isclose(zero, 0.5, rel_tol=1e-6)
    assert positive > zero
    assert negative < zero


def test_aggregate_expected_score_weights_by_percentage():
    moves = [
        {"sf_eval_cp": 100, "adjusted_pct": 60.0},
        {"sf_eval_cp": -50, "adjusted_pct": 40.0},
    ]

    expected = aggregate_expected_score(moves)

    assert 0.0 <= expected <= 1.0

    expected_manual = (
        expected_score_cp(100) * 0.6 + expected_score_cp(-50) * 0.4
    )
    assert math.isclose(expected, expected_manual, rel_tol=1e-6)


def test_expected_score_handles_mate():
    moves = [{"sf_eval_mate": 2, "adjusted_pct": 100.0}]

    expected = aggregate_expected_score(moves)
    assert expected > 0.99
