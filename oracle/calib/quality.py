"""Move quality classification helpers."""

from __future__ import annotations

import os
from typing import Mapping

import yaml

DEFAULT_THRESHOLDS = {
    "good": 20.0,
    "inaccuracy": 60.0,
    "mistake": 200.0,
    "blunder": 500.0,
}

ENV_MAPPING = {
    "good": "ORACLE_QUALITY_GOOD",
    "inaccuracy": "ORACLE_QUALITY_INACCURACY",
    "mistake": "ORACLE_QUALITY_MISTAKE",
    "blunder": "ORACLE_QUALITY_BLUNDER",
}

ORDER = ["good", "inaccuracy", "mistake", "blunder"]


def load_thresholds(
    env: Mapping[str, str] | None = None,
    yaml_path: str | None = None,
) -> Mapping[str, float]:
    """Load quality thresholds from environment variables or YAML."""

    env_map = env if env is not None else os.environ
    thresholds = {key: float(value) for key, value in DEFAULT_THRESHOLDS.items()}

    for key, env_key in ENV_MAPPING.items():
        value = env_map.get(env_key)
        if value is not None:
            thresholds[key] = float(value)

    if yaml_path:
        with open(yaml_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        quality_section = data.get("quality", data)
        for key in ORDER:
            if key in quality_section:
                thresholds[key] = float(quality_section[key])

    previous = 0.0
    for key in ORDER:
        thresholds[key] = max(previous, thresholds[key])
        previous = thresholds[key]

    return thresholds


def classify_quality(
    best_cp: float | None,
    candidate_cp: float | None,
    thresholds: Mapping[str, float],
) -> str:
    """Classify a move quality bucket based on evaluation delta."""

    if best_cp is None or candidate_cp is None:
        return "unknown"

    delta = max(0.0, best_cp - candidate_cp)

    if delta <= thresholds.get("good", 0.0):
        return "good"
    if delta <= thresholds.get("inaccuracy", thresholds.get("good", 0.0)):
        return "inaccuracy"
    if delta <= thresholds.get("mistake", thresholds.get("inaccuracy", 0.0)):
        return "mistake"
    return "blunder"

