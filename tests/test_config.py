"""Tests for configuration loading helpers."""

from __future__ import annotations

from oracle import config


def test_load_settings_recovers_from_invalid_numbers(monkeypatch):
    """Invalid numeric env vars should fall back to defaults."""

    monkeypatch.setenv("STOCKFISH_TIME_LIMIT", "not-a-number")
    monkeypatch.setenv("STOCKFISH_DEPTH", "twenty")
    monkeypatch.setenv("STOCKFISH_THREADS", "  ")
    monkeypatch.setenv("STOCKFISH_HASH", "n/a")

    settings = config.load_settings()

    assert settings.time_limit == 1.3
    assert settings.depth == 20
    assert settings.threads == 8
    assert settings.hash_size == 512


def test_load_settings_accepts_zero_values(monkeypatch):
    """Zero values are valid inputs and should be preserved."""

    monkeypatch.setenv("STOCKFISH_TIME_LIMIT", "0")
    monkeypatch.setenv("STOCKFISH_DEPTH", "0")
    monkeypatch.setenv("STOCKFISH_THREADS", "0")
    monkeypatch.setenv("STOCKFISH_HASH", "0")

    settings = config.load_settings()

    assert settings.time_limit == 0.0
    assert settings.depth == 0
    assert settings.threads == 0
    assert settings.hash_size == 0
