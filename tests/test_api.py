"""Unit tests for serving API (RF-06 / RF-07)."""

from __future__ import annotations

from pathlib import Path

import joblib
from fastapi.testclient import TestClient

from api.main import create_app
from api.predictor import (
    ABOVE_MARKET,
    FAIR_PRICE,
    GOOD_DEAL,
    ModelPredictor,
)
from ml.train import PipelineBuilder
import pandas as pd


def _tiny_model(tmp_path: Path) -> Path:
    rows = [
        {
            "surface_m2": 50.0 + i,
            "rooms": 2,
            "distance_from_center_km": 3.0,
            "area_price_per_m2_hist": 22.0,
            "publication_month": 9,
            "municipio": "I" if i % 2 == 0 else "II",
            "y": 20.0 + i * 0.3,
        }
        for i in range(20)
    ]
    X = pd.DataFrame(
        [
            {
                "surface_m2": r["surface_m2"],
                "rooms": r["rooms"],
                "distance_from_center_km": r["distance_from_center_km"],
                "area_price_per_m2_hist": r["area_price_per_m2_hist"],
                "publication_month": r["publication_month"],
                "municipio": r["municipio"],
            }
            for r in rows
        ]
    )
    y = [r["y"] for r in rows]
    pipe = PipelineBuilder().build()
    pipe.fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    return path


def test_classify_deal_bands():
    assert ModelPredictor.classify_deal(18.0, 20.0) == GOOD_DEAL
    assert ModelPredictor.classify_deal(20.0, 20.0) == FAIR_PRICE
    assert ModelPredictor.classify_deal(22.5, 20.0) == ABOVE_MARKET


def test_predictor_score(tmp_path: Path):
    model_path = _tiny_model(tmp_path)
    pred = ModelPredictor(model_path)
    out = pred.score(
        {
            "surface_m2": 55.0,
            "rooms": 2,
            "distance_from_center_km": 3.0,
            "area_price_per_m2_hist": 22.0,
            "publication_month": 9,
            "municipio": "I",
        },
        actual_price_per_m2=15.0,
    )
    assert out["predicted_price_per_m2_monthly"] > 0
    assert out["deal_label"] in {GOOD_DEAL, FAIR_PRICE, ABOVE_MARKET}
    assert "gap_pct" in out


def test_predict_endpoint(tmp_path: Path):
    model_path = _tiny_model(tmp_path)
    with TestClient(create_app(model_path)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        resp = client.post(
            "/predict",
            json={
                "surface_m2": 55.0,
                "rooms": 2,
                "distance_from_center_km": 3.0,
                "area_price_per_m2_hist": 22.0,
                "publication_month": 9,
                "municipio": "I",
                "price_per_m2_monthly": 15.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["predicted_price_per_m2_monthly"] > 0
        assert body["deal_label"] is not None


def test_health_degraded_without_model(tmp_path: Path):
    missing = tmp_path / "missing.joblib"
    with TestClient(create_app(missing)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is False
        assert health.json()["status"] == "degraded"
        resp = client.post(
            "/predict",
            json={
                "surface_m2": 55.0,
                "rooms": 2,
                "publication_month": 9,
                "municipio": "I",
            },
        )
        assert resp.status_code == 503
