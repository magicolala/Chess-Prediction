"""Normalization helpers for probability distributions."""

from __future__ import annotations

import math
from typing import List, Tuple


def normalize_to_pct(move_logprobs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Normalize log probabilities into percentage values."""

    if not move_logprobs:
        return []

    log_values = [value for _, value in move_logprobs]
    max_log = max(log_values)
    exp_sum = sum(math.exp(value - max_log) for value in log_values)
    factor = 100.0 / exp_sum

    normalized: List[Tuple[str, float]] = []
    for move, value in move_logprobs:
        probability = math.exp(value - max_log) * factor
        normalized.append((move, probability))
    return normalized

