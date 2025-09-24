from oracle.config import Settings
from oracle.domain.models import MovePrediction, PredictionReport
import oracle.interfaces.web.app as web_app


def _dummy_report() -> PredictionReport:
    return PredictionReport(
        rows=[
            MovePrediction(
                move="Nf3",
                raw_probability=0.1,
                normalized_probability=55.0,
                win_percentage=60.0,
                percentage_loss=5.0,
            )
        ],
        current_win_percentage=60.0,
        elapsed_seconds=0.2,
    )


def test_stockfish_path_override_in_form(monkeypatch):
    created_paths: list[str] = []

    def fake_create_single_move_predictor(settings: Settings):
        created_paths.append(settings.stockfish_path)

        class _Predictor:
            def predict(self, context, depth, prob_threshold):  # noqa: ANN001
                return _dummy_report()

        return _Predictor()

    base_settings = Settings(
        openai_api_key="key",
        stockfish_path="/default/stockfish",
        model_name="model",
        time_limit=0.5,
        depth=3,
        threads=1,
        hash_size=128,
    )

    monkeypatch.setattr(
        web_app, "create_single_move_predictor", fake_create_single_move_predictor
    )
    monkeypatch.setattr(web_app, "load_settings", lambda: base_settings)

    app = web_app.create_app()
    client = app.test_client()

    response = client.post(
        "/",
        data={
            "pgn": "1. e4 e5 2. Nf3 Nc6",
            "white_elo": "2000",
            "black_elo": "2100",
            "game_type": "rapid",
            "depth": "3",
            "prob_threshold": "0.01",
            "stockfish_path": "/custom/stockfish",
        },
    )

    assert response.status_code == 200
    assert "/custom/stockfish" in created_paths
    assert "/default/stockfish" not in created_paths


def test_stockfish_path_falls_back_to_settings(monkeypatch):
    created_paths: list[str] = []

    def fake_create_single_move_predictor(settings: Settings):
        created_paths.append(settings.stockfish_path)

        class _Predictor:
            def predict(self, context, depth, prob_threshold):  # noqa: ANN001
                return _dummy_report()

        return _Predictor()

    base_settings = Settings(
        openai_api_key="key",
        stockfish_path="/configured/stockfish",
        model_name="model",
        time_limit=0.5,
        depth=3,
        threads=1,
        hash_size=128,
    )

    monkeypatch.setattr(
        web_app, "create_single_move_predictor", fake_create_single_move_predictor
    )
    monkeypatch.setattr(web_app, "load_settings", lambda: base_settings)

    app = web_app.create_app()
    client = app.test_client()

    response = client.post(
        "/",
        data={
            "pgn": "1. e4 e5 2. Nf3 Nc6",
            "white_elo": "2000",
            "black_elo": "2100",
            "game_type": "rapid",
            "depth": "3",
            "prob_threshold": "0.01",
            "stockfish_path": "",
        },
    )

    assert response.status_code == 200
    assert created_paths == ["/configured/stockfish"]


def test_missing_stockfish_path_shows_error(monkeypatch):
    base_settings = Settings(
        openai_api_key="key",
        stockfish_path="",
        model_name="model",
        time_limit=0.5,
        depth=3,
        threads=1,
        hash_size=128,
    )

    monkeypatch.setattr(web_app, "load_settings", lambda: base_settings)

    def _fail(_settings):  # noqa: ANN001
        raise AssertionError("Predictor should not be created")

    monkeypatch.setattr(web_app, "create_single_move_predictor", _fail)

    app = web_app.create_app()
    client = app.test_client()

    response = client.post(
        "/",
        data={
            "pgn": "1. e4 e5 2. Nf3 Nc6",
            "white_elo": "2000",
            "black_elo": "2100",
            "game_type": "rapid",
            "depth": "3",
            "prob_threshold": "0.01",
            "stockfish_path": "",
        },
    )

    assert response.status_code == 200
    assert "chemin de Stockfish" in response.get_data(as_text=True)
