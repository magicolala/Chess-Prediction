"""llama.cpp-based local LLM backend."""

from __future__ import annotations

import math
from typing import List, Tuple

from .base import SequenceProvider

try:  # pragma: no cover - optional dependency import
    from llama_cpp import Llama
except Exception:  # pragma: no cover - handled gracefully
    Llama = None  # type: ignore[misc]


def _log_softmax_row(row: List[float]) -> List[float]:
    max_val = max(row)
    exp_sum = sum(math.exp(value - max_val) for value in row)
    return [value - max_val - math.log(exp_sum) for value in row]


class LlamaCppLocalProvider(SequenceProvider):
    """Sequence provider backed by a llama.cpp model."""

    llama_class = Llama

    def __init__(self, model_path: str, llama=None) -> None:
        if llama is None:
            if self.llama_class is None:
                raise ImportError(
                    "llama_cpp must be installed to use LlamaCppLocalProvider"
                )
            llama = self.llama_class(model_path=model_path, logits_all=True)
        self.model_path = model_path
        self._llama = llama

    def get_top_sequences(
        self,
        prompt: str,
        legal_moves: List[str],
        depth: int,
        prob_threshold: float,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        prompt_bytes = prompt.encode("utf-8")
        tokens = self._llama.tokenize(prompt_bytes, add_bos=True)
        self._llama.eval(tokens)
        logits = self._llama.get_logits()
        log_probs = _log_softmax_row(list(logits))
        indexed = list(enumerate(log_probs))
        indexed.sort(key=lambda item: item[1], reverse=True)

        results: List[Tuple[str, float]] = []
        for token_id, logprob in indexed:
            token_bytes = self._llama.detokenize([token_id])
            token_text = token_bytes.decode("utf-8").strip()
            if token_text:
                results.append((token_text, float(logprob)))
            if len(results) >= top_k:
                break

        return results

