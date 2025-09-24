import json

import Oracle_web as web_app


def test_api_analyze_returns_expected_json():
    def fake_analyze(**kwargs):
        assert kwargs["pgn"].startswith("1. e4")
        assert kwargs["ctx"] == {"elo": 1500, "time_control": "rapid"}
        return {
            "model": "fake",
            "expected_score": 0.75,
            "usage": {"top_k": 2},
            "moves": [
                {
                    "san": "Nf3",
                    "prior_pct": 55.0,
                    "adjusted_pct": 60.0,
                    "sf_eval_cp": 80,
                    "quality": "good",
                }
            ],
        }

    app = web_app.create_app(analyze_fn=fake_analyze)
    client = app.test_client()

    payload = {
        "pgn": "1. e4 e5",
        "context": {"elo": 1500, "time_control": "rapid"},
    }
    response = client.post(
        "/api/analyze",
        data=json.dumps(payload),
        content_type="application/json",
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["model"] == "fake"
    assert data["expected_score"] == 0.75
    assert data["moves"][0]["san"] == "Nf3"
    assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"


def test_api_analyze_accepts_stockfish_path(monkeypatch):
    captured_paths = []

    class DummyEngine:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def quit(self):
            pass

    def fake_make_engine_factory(path):
        def factory():
            captured_paths.append(path)
            return DummyEngine()

        return factory

    def fake_analyze(**kwargs):
        engine_factory = kwargs.get("engine_factory")
        assert engine_factory is not None
        with engine_factory():
            pass
        return {
            "model": "fake",
            "expected_score": 0.5,
            "usage": {"top_k": 2},
            "moves": [],
        }

    monkeypatch.setattr(web_app, "_make_engine_factory", fake_make_engine_factory)
    app = web_app.create_app(analyze_fn=fake_analyze)
    client = app.test_client()

    payload = {
        "pgn": "1. e4 e5",
        "context": {"elo": 1500, "time_control": "rapid"},
        "stockfish_path": "/custom/stockfish",
    }
    response = client.post(
        "/api/analyze",
        data=json.dumps(payload),
        content_type="application/json",
    )

    assert response.status_code == 200
    assert captured_paths == ["/custom/stockfish"]
