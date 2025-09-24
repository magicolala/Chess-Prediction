import math

import pytest

from oracle.calib.calibrator import Calibrator


def test_calibrator_boosts_good_moves():
    config = {
        "quality_bias": {
            "good": 0.4,
            "mistake": -0.4,
            "blunder": -0.8,
            "unknown": 0.0,
        },
        "time_control_bias": {
            "rapid": 0.0,
        },
        "elo": {
            "reference": 1500,
            "scale": 0.0,
        },
    }

    calibrator = Calibrator(config=config)

    moves = [
        {"san": "e4", "prior_logprob": math.log(0.6), "quality": "good"},
        {"san": "d4", "prior_logprob": math.log(0.4), "quality": "mistake"},
    ]

    results = calibrator.adjust(
        moves,
        context={"elo": 1500, "time_control": "rapid"},
    )

    prior_pct = {item["san"]: item["prior_pct"] for item in results}
    adjusted_pct = {item["san"]: item["adjusted_pct"] for item in results}

    assert pytest.approx(sum(adjusted_pct.values()), abs=0.1) == 100.0
    assert adjusted_pct["e4"] > prior_pct["e4"]
    assert adjusted_pct["d4"] < prior_pct["d4"]


def test_calibrator_accounts_for_elo_and_time():
    config = {
        "quality_bias": {"good": 0.0, "unknown": 0.0},
        "time_control_bias": {"bullet": -0.2, "classical": 0.2},
        "elo": {"reference": 1500, "scale": 0.001},
    }
    calibrator = Calibrator(config=config)

    moves = [
        {"san": "e4", "prior_logprob": math.log(0.5), "quality": "unknown"},
        {"san": "d4", "prior_logprob": math.log(0.5), "quality": "unknown"},
    ]

    low_elo = calibrator.adjust(
        moves,
        context={"elo": 1000, "time_control": "bullet"},
    )
    high_elo = calibrator.adjust(
        moves,
        context={"elo": 2000, "time_control": "classical"},
    )

    low_adjusted = {item["san"]: item["adjusted_pct"] for item in low_elo}
    high_adjusted = {item["san"]: item["adjusted_pct"] for item in high_elo}

    assert low_adjusted["e4"] == pytest.approx(low_adjusted["d4"], rel=1e-6)
    assert high_adjusted["e4"] == pytest.approx(high_adjusted["d4"], rel=1e-6)

    assert sum(item["adjusted_pct"] for item in low_elo) == pytest.approx(100.0, abs=0.1)
    assert sum(item["adjusted_pct"] for item in high_elo) == pytest.approx(100.0, abs=0.1)
