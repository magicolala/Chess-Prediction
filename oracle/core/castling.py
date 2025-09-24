"""Utilities to disambiguate castling moves."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


_MIN_PROBABILITY = 1e-12


def _safe_log(probability: float) -> float:
    return math.log(max(probability, _MIN_PROBABILITY))


def split_castling(
    move: str, base_logprob: float, continuation_logprob: Optional[float]
) -> List[Tuple[str, float]]:
    """Split a castling candidate into short and long variants."""

    if move != "O-O" or continuation_logprob is None:
        return [(move, base_logprob)]

    base_prob = math.exp(base_logprob)
    continuation_prob = math.exp(continuation_logprob)
    continuation_prob = max(0.0, min(1.0, continuation_prob))

    long_prob = base_prob * continuation_prob
    short_prob = base_prob - long_prob
    short_prob = max(short_prob, 0.0)

    return [("O-O", _safe_log(short_prob)), ("O-O-O", _safe_log(long_prob))]

