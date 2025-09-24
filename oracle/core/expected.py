"""Expected score computations from engine evaluations."""

from __future__ import annotations

import math
from typing import Iterable, Mapping


def expected_score_cp(cp: float, k: float = 0.0035) -> float:
    """Convert centipawn evaluation to expected score using logistic."""

    return 1.0 / (1.0 + math.exp(-k * cp))


def _score_from_move(move: Mapping[str, float]) -> float:
    if "sf_eval_mate" in move and move["sf_eval_mate"] is not None:
        mate_value = float(move["sf_eval_mate"])
        cp_equivalent = 10000.0 if mate_value > 0 else -10000.0
        return expected_score_cp(cp_equivalent)
    cp = float(move.get("sf_eval_cp", 0.0))
    return expected_score_cp(cp)


def aggregate_expected_score(moves: Iterable[Mapping[str, float]]) -> float:
    """Compute weighted expected score across moves."""

    total = 0.0
    for move in moves:
        pct = float(move.get("adjusted_pct", 0.0)) / 100.0
        if pct <= 0:
            continue
        total += pct * _score_from_move(move)
    return total

