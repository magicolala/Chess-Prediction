"""Calibrate LLM move probabilities with chess heuristics."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping

import yaml

from oracle.core.normalize import normalize_to_pct

DEFAULT_CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__), "calibration.default.yaml"
)


def load_config(yaml_path: str | None = None) -> Dict[str, Any]:
    """Load calibration configuration from YAML."""

    path = yaml_path or os.getenv("ORACLE_CALIBRATION_YAML", DEFAULT_CALIBRATION_PATH)
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class Calibrator:
    """Adjust move probabilities based on external heuristics."""

    def __init__(self, config: Mapping[str, Any] | None = None, yaml_path: str | None = None):
        if config is None:
            config = load_config(yaml_path)
        self.config = config

    def _quality_bias(self, quality: str | None) -> float:
        mapping = self.config.get("quality_bias", {})
        return float(mapping.get(quality or "unknown", mapping.get("unknown", 0.0)))

    def _time_bias(self, time_control: str | None) -> float:
        mapping = self.config.get("time_control_bias", {})
        if time_control is None:
            return 0.0
        return float(mapping.get(time_control, 0.0))

    def _elo_bias(self, elo: int | None) -> float:
        elo_cfg = self.config.get("elo", {})
        reference = float(elo_cfg.get("reference", 1500))
        scale = float(elo_cfg.get("scale", 0.0))
        if elo is None:
            return 0.0
        return (float(elo) - reference) * scale

    def adjust(
        self,
        moves: Iterable[Mapping[str, Any]],
        context: Mapping[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Return calibrated move probabilities."""

        move_list = list(moves)
        if not move_list:
            return []
        context = context or {}

        prior_pairs = [
            (move["san"], float(move["prior_logprob"])) for move in move_list
        ]
        prior_pct = dict(normalize_to_pct(prior_pairs))

        adjusted_pairs = []
        biases: Dict[str, float] = {}
        for move in move_list:
            bias = self._quality_bias(move.get("quality"))
            bias += self._time_bias(context.get("time_control"))
            bias += self._elo_bias(context.get("elo"))
            biases[move["san"]] = bias
            adjusted_pairs.append((move["san"], float(move["prior_logprob"]) + bias))

        adjusted_pct = dict(normalize_to_pct(adjusted_pairs))

        results: List[Dict[str, Any]] = []
        for move in move_list:
            san = move["san"]
            results.append(
                {
                    "san": san,
                    "prior_logprob": float(move["prior_logprob"]),
                    "prior_pct": prior_pct[san],
                    "bias": biases[san],
                    "adjusted_logprob": float(move["prior_logprob"]) + biases[san],
                    "adjusted_pct": adjusted_pct[san],
                    "quality": move.get("quality"),
                }
            )
        return results

