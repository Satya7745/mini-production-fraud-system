from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from serving.app import app
from serving.predictor import Predictor
from src.data_pipeline import FEATURES

ROOT = Path(__file__).resolve().parents[1]


def payload() -> dict[str, float]:
    row = pd.read_csv(ROOT / "data" / "sample_transactions.csv", nrows=1, usecols=FEATURES).iloc[0]
    return {name: float(row[name]) for name in FEATURES}


def test_health_and_prediction_contract():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["artifact_loaded"] is True
        response = client.post("/predict", json=payload())
        assert response.status_code == 200
        body = response.json()
        assert 0 <= body["fraud_probability"] <= 1
        assert body["fraud_prediction"] in (0, 1)
        assert body["fallback_used"] is False
        assert response.headers["x-request-id"]


def test_schema_rejects_missing_negative_and_extra_fields():
    valid = payload()
    with TestClient(app) as client:
        missing = valid.copy(); missing.pop("V28")
        assert client.post("/predict", json=missing).status_code == 422
        negative = valid.copy(); negative["Amount"] = -1.0
        assert client.post("/predict", json=negative).status_code == 422
        extra = {**valid, "future_label": 1}
        assert client.post("/predict", json=extra).status_code == 422


def test_predictor_returns_prior_fallback_on_scoring_failure():
    predictor = Predictor(ROOT / "artifacts")

    class BrokenModel:
        def predict_proba(self, frame):
            raise RuntimeError("injected failure")

    predictor.model = BrokenModel()
    result = predictor.score(payload())
    assert result.fallback_used is True
    assert result.fraud_probability == predictor.prior
