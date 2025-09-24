"""Expansion utilities to build SAN moves from token prefixes."""

from __future__ import annotations

import math
from typing import List, Set, Tuple

from oracle.llm.base import SequenceProvider


def expand_san_candidates(
    provider: SequenceProvider,
    prompt: str,
    legal_moves: Set[str],
    depth: int,
    prob_threshold: float,
    top_k: int,
) -> List[Tuple[str, float]]:
    """Expand token prefixes into legal SAN candidates."""

    if not legal_moves or depth <= 0:
        return []

    min_logprob = math.log(prob_threshold) if prob_threshold > 0 else float("-inf")

    results: List[Tuple[str, float]] = []

    initial_candidates = provider.get_top_sequences(
        prompt=prompt,
        legal_moves=list(legal_moves),
        depth=depth,
        prob_threshold=prob_threshold,
        top_k=top_k,
    )

    def _expand(prefix: str, logprob: float, remaining_depth: int) -> None:
        if prefix in legal_moves:
            results.append((prefix, logprob))
        if remaining_depth <= 0:
            return
        next_prompt = f"{prompt}{prefix}"
        next_candidates = provider.get_top_sequences(
            prompt=next_prompt,
            legal_moves=list(legal_moves),
            depth=remaining_depth,
            prob_threshold=prob_threshold,
            top_k=top_k,
        )
        for token_text, token_logprob in next_candidates:
            candidate = f"{prefix}{token_text}"
            if not any(move.startswith(candidate) for move in legal_moves):
                continue
            new_logprob = logprob + token_logprob
            if new_logprob < min_logprob:
                continue
            _expand(candidate, new_logprob, remaining_depth - 1)

    for token_text, token_logprob in initial_candidates:
        if not any(move.startswith(token_text) for move in legal_moves):
            continue
        if token_logprob < min_logprob:
            continue
        _expand(token_text, token_logprob, depth - 1)

    return results

