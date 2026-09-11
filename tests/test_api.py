"""Unit tests for serving API (RF-06 / RF-07)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from api.predictor import (
    ABOVE_OMI_BAND,
    BELOW_OMI_BAND,
    IN_BAND,
    ModelPredictor,
)
from ml.train import FEATURE_COLS, PipelineBuilder
from tests.omi_rows import omi_feature_row


def _tiny_model(tmp_path: Path) -> Path:
    rows = [omi_feature_row(i, day="2026-09-08", loc_mid_lag=15.0 + i) for i in range(24)]
    X = pd.DataFrame([{c: r.get(c) for c in FEATURE_COLS} for r in rows])
    y = [float(r["price_per_m2_monthly"]) for r in rows]
    pipe = PipelineBuilder().build()
    pipe.fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    return path


def test_classify_deal_bands():
    assert ModelPredictor.classify_deal(18.0, 20.0) == BELOW_OMI_BAND
    assert ModelPredictor.classify_deal(20.0, 20.0) == IN_BAND
    assert ModelPredictor.classify_deal(22.5, 20.0) == ABOVE_OMI_BAND


def test_predictor_score(tmp_path: Path):
    model_path = _tiny_model(tmp_path)
    pred = ModelPredictor(model_path)
    out = pred.score(
        {
            "zona_omi": "B12",
            "tipologia": "Abitazioni civili",
            "stato": "NORMALE",
            "publication_month": 12,
            "loc_mid_lag": 18.0,
        },
        actual_price_per_m2=10.0,
    )
    assert out["predicted_price_per_m2_monthly"] > 0
    assert out["deal_label"] in {BELOW_OMI_BAND, IN_BAND, ABOVE_OMI_BAND}
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
                "zona_omi": "B12",
                "tipologia": "Abitazioni civili",
                "stato": "NORMALE",
                "publication_month": 12,
                "loc_mid_lag": 18.0,
                "price_per_m2_monthly": 10.0,
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
                "zona_omi": "B12",
                "tipologia": "Abitazioni civili",
                "stato": "NORMALE",
                "publication_month": 12,
            },
        )
        assert resp.status_code == 503
