from textwrap import dedent
import importlib.machinery
import types

import pytest


def _load_module(path: str, name: str):
    loader = importlib.machinery.SourceFileLoader(name, path)
    module = types.ModuleType(name)
    loader.exec_module(module)
    return module


cli_one_move = _load_module("Oracle_one_move", "oracle_one_move")


@pytest.fixture
def sample_pgn():
    return dedent(
        """
        [Event "Test"]

        1. e4 e5
        """
    ).strip()


def test_one_move_cli_outputs_table(sample_pgn, capsys):
    def fake_analyze(**kwargs):
        return {
            "model": "fake",
            "expected_score": 0.75,
            "usage": {"top_k": kwargs.get("top_k", 2)},
            "moves": [
                {
                    "san": "Nf3",
                    "prior_pct": 55.0,
                    "adjusted_pct": 60.0,
                    "sf_eval_cp": 80,
                    "quality": "good",
                },
                {
                    "san": "Nc3",
                    "prior_pct": 45.0,
                    "adjusted_pct": 40.0,
                    "sf_eval_cp": 10,
                    "quality": "mistake",
                },
            ],
        }

    result = cli_one_move.main(
        pgn=sample_pgn,
        analyze_fn=fake_analyze,
        depth=2,
        prob_threshold=0.0,
        top_k=2,
    )

    captured = capsys.readouterr()
    output = captured.out
    assert "Move" in output
    assert "Prior %" in output
    assert "Adjusted %" in output
    assert "Nf3" in output
    assert "Nc3" in output
    assert result["expected_score"] == 0.75
