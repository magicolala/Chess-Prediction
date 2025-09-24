"""Transformers-based local LLM backend."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .base import SequenceProvider

try:  # pragma: no cover - optional dependency import
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - handled in code
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


def _to_numpy(data):
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        return data.numpy()
    return data


def _log_softmax_row(row: List[float]) -> List[float]:
    max_val = max(row)
    exp_sum = sum(math.exp(value - max_val) for value in row)
    return [value - max_val - math.log(exp_sum) for value in row]


class TransformersLocalProvider(SequenceProvider):
    """Sequence provider using a local transformers model."""

    def __init__(
        self,
        model_id: str,
        tokenizer=None,
        model=None,
        device: Optional[str] = None,
    ) -> None:
        if tokenizer is None or model is None:
            if AutoTokenizer is None or AutoModelForCausalLM is None:
                raise ImportError(
                    "transformers must be installed to use TransformersLocalProvider"
                )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
            if device:
                model.to(device)
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.model = model

    def get_top_sequences(
        self,
        prompt: str,
        legal_moves: List[str],
        depth: int,
        prob_threshold: float,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        generate_kwargs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded.get("attention_mask"),
            "max_new_tokens": 1,
            "output_scores": True,
            "return_dict_in_generate": True,
            "do_sample": False,
            "temperature": 0,
        }
        outputs = self.model.generate(**generate_kwargs)
        scores = outputs.scores[0]
        score_array = _to_numpy(scores)[0]

        log_probs = _log_softmax_row(list(score_array))
        indexed = list(enumerate(log_probs))
        indexed.sort(key=lambda item: item[1], reverse=True)

        results: List[Tuple[str, float]] = []
        for token_id, logprob in indexed[: max(top_k, 0)]:
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            token_text = token_text.strip()
            if token_text:
                results.append((token_text, float(logprob)))
            if len(results) >= top_k:
                break

        return results

