import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None
else:
    load_dotenv()


DEFAULT_MODEL = "gpt-3.5-turbo-instruct"


@dataclass
class Settings:
    openai_api_key: str = ""
    stockfish_path: str = ""
    model_name: str = DEFAULT_MODEL
    time_limit: float = 1.3
    depth: int = 20
    threads: int = 8
    hash_size: int = 512


def _coerce_float_env(var_name: str, default: float) -> float:
    """Return a float environment variable or a default on failure."""

    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default

    stripped = raw_value.strip()
    if stripped == "":
        return default

    try:
        return float(stripped)
    except ValueError:
        return default


def _coerce_int_env(var_name: str, default: int) -> int:
    """Return an int environment variable or a default on failure."""

    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default

    stripped = raw_value.strip()
    if stripped == "":
        return default

    try:
        return int(stripped)
    except ValueError:
        return default


def load_settings() -> Settings:
    """Load configuration from environment variables."""

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        stockfish_path=os.getenv("STOCKFISH_PATH", ""),
        model_name=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        time_limit=_coerce_float_env("STOCKFISH_TIME_LIMIT", 1.3),
        depth=_coerce_int_env("STOCKFISH_DEPTH", 20),
        threads=_coerce_int_env("STOCKFISH_THREADS", 8),
        hash_size=_coerce_int_env("STOCKFISH_HASH", 512),
    )


def resolve_value(explicit: Optional[str], fallback: str) -> str:
    return explicit if explicit not in (None, "") else fallback


__all__ = [
    "Settings",
    "load_settings",
    "resolve_value",
    "DEFAULT_MODEL",
]
