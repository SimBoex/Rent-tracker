"""Unit tests for serving API (RF-06 / RF-07)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from api.omi_band import band_payload, build_band_index
from api.predictor import (
    ABOVE_OMI_BAND,
    BELOW_OMI_BAND,
    DEAL_BASIS_MODEL,
    DEAL_BASIS_OMI,
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
    assert ModelPredictor.classify_deal(18.0, 20.0) == (BELOW_OMI_BAND, DEAL_BASIS_MODEL)
    assert ModelPredictor.classify_deal(20.0, 20.0) == (IN_BAND, DEAL_BASIS_MODEL)
    assert ModelPredictor.classify_deal(22.5, 20.0) == (ABOVE_OMI_BAND, DEAL_BASIS_MODEL)


def test_classify_deal_omi_band():
    assert ModelPredictor.classify_deal(
        13.0, 20.0, omi_loc_min=14.0, omi_loc_max=18.0
    ) == (BELOW_OMI_BAND, DEAL_BASIS_OMI)
    assert ModelPredictor.classify_deal(
        16.0, 20.0, omi_loc_min=14.0, omi_loc_max=18.0
    ) == (IN_BAND, DEAL_BASIS_OMI)
    assert ModelPredictor.classify_deal(
        19.0, 20.0, omi_loc_min=14.0, omi_loc_max=18.0
    ) == (ABOVE_OMI_BAND, DEAL_BASIS_OMI)


def test_band_payload_half_width():
    band = band_payload(14.0, 18.0)
    assert band["omi_loc_min"] == 14.0
    assert band["omi_loc_max"] == 18.0
    assert band["omi_half_width"] == 2.0
    assert band["band_source"] == "omi"


def test_build_band_index_keeps_latest_semester(tmp_path: Path):
    path = tmp_path / "features.jsonl"
    rows = [
        {
            "zona_omi": "B12",
            "tipologia": "Abitazioni civili",
            "stato": "NORMALE",
            "semester": "2024-1",
            "omi_loc_min": 10.0,
            "omi_loc_max": 12.0,
        },
        {
            "zona_omi": "B12",
            "tipologia": "Abitazioni civili",
            "stato": "NORMALE",
            "semester": "2025-1",
            "omi_loc_min": 14.0,
            "omi_loc_max": 18.0,
        },
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    index = build_band_index(path)
    band = index[("B12", "Abitazioni civili", "NORMALE")]
    assert band["omi_loc_min"] == 14.0
    assert band["omi_half_width"] == 2.0


def test_predictor_score(tmp_path: Path):
    model_path = _tiny_model(tmp_path)
    pred = ModelPredictor(model_path, features_path=None)
    out = pred.score(
        {
            "zona_omi": "B12",
            "tipologia": "Abitazioni civili",
            "stato": "NORMALE",
            "loc_mid_lag": 18.0,
            "omi_loc_min": 14.0,
            "omi_loc_max": 18.0,
        },
        actual_price_per_m2=10.0,
    )
    assert out["predicted_price_per_m2_monthly"] > 0
    assert out["deal_label"] == BELOW_OMI_BAND
    assert out["deal_basis"] == DEAL_BASIS_OMI
    assert "gap_pct" in out
    assert out["omi_loc_min"] == 14.0
    assert out["omi_loc_max"] == 18.0
    assert out["omi_half_width"] == 2.0
    assert out["band_source"] == "omi"
    assert out["shap_values"]
    assert {row["feature"] for row in out["shap_values"]} == set(FEATURE_COLS)
    assert out["shap_base_value"] is not None


def test_predictor_score_skip_shap(tmp_path: Path):
    model_path = _tiny_model(tmp_path)
    pred = ModelPredictor(model_path, features_path=None)
    out = pred.score(
        {
            "zona_omi": "B12",
            "tipologia": "Abitazioni civili",
            "stato": "NORMALE",
            "loc_mid_lag": 18.0,
        },
        include_shap=False,
    )
    assert out["shap_values"] is None
    assert out["shap_base_value"] is None


def test_predictor_score_lookup_from_features(tmp_path: Path):
    model_path = _tiny_model(tmp_path)
    feats = tmp_path / "features.jsonl"
    feats.write_text(
        json.dumps(
            {
                "zona_omi": "B12",
                "tipologia": "Abitazioni civili",
                "stato": "NORMALE",
                "semester": "2025-2",
                "omi_loc_min": 16.0,
                "omi_loc_max": 20.0,
                "price_per_m2_monthly": 18.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pred = ModelPredictor(model_path, features_path=feats)
    out = pred.score(
        {
            "zona_omi": "B12",
            "tipologia": "Abitazioni civili",
            "stato": "NORMALE",
            "loc_mid_lag": 18.0,
        }
    )
    assert out["omi_loc_min"] == 16.0
    assert out["omi_half_width"] == 2.0


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
                "loc_mid_lag": 18.0,
                "price_per_m2_monthly": 10.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["predicted_price_per_m2_monthly"] > 0
        assert body["deal_label"] is not None
        assert body["deal_basis"] in {"omi_band", "model_pct"}
        assert "omi_half_width" in body
        assert body["shap_values"]
        assert body["shap_base_value"] is not None


def test_profile_history_endpoint(tmp_path: Path):
    model_path = _tiny_model(tmp_path)
    feats = tmp_path / "features.jsonl"
    rows = [
        {
            "zona_omi": "B12",
            "tipologia": "Abitazioni civili",
            "stato": "NORMALE",
            "semester": "2024-2",
            "price_per_m2_monthly": 16.0,
            "omi_loc_min": 14.0,
            "omi_loc_max": 18.0,
            "loc_mid_lag": 15.0,
        },
        {
            "zona_omi": "B12",
            "tipologia": "Abitazioni civili",
            "stato": "NORMALE",
            "semester": "2025-1",
            "price_per_m2_monthly": 17.0,
            "omi_loc_min": 15.0,
            "omi_loc_max": 19.0,
            "loc_mid_lag": 16.0,
        },
        {
            "zona_omi": "C1",
            "tipologia": "Abitazioni civili",
            "stato": "NORMALE",
            "semester": "2025-1",
            "price_per_m2_monthly": 12.0,
            "omi_loc_min": 10.0,
            "omi_loc_max": 14.0,
        },
    ]
    feats.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    with TestClient(create_app(model_path, features_path=feats)) as client:
        resp = client.get(
            "/profile/history",
            params={
                "zona_omi": "B12",
                "tipologia": "Abitazioni civili",
                "stato": "NORMALE",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["test_semester"] == "2025-1"
        assert len(body["series"]) == 2
        assert body["series"][0]["role"] == "train"
        assert body["series"][1]["role"] == "test"
        assert body["series"][1]["price_per_m2_monthly"] == 17.0
        assert body["next_prediction"]["loc_mid_lag"] == 17.0
        assert body["next_prediction"]["predicted_price_per_m2_monthly"] > 0

        missing = client.get(
            "/profile/history",
            params={
                "zona_omi": "ZZ99",
                "tipologia": "Abitazioni civili",
                "stato": "NORMALE",
            },
        )
        assert missing.status_code == 404


def test_profile_history_503_without_features(tmp_path: Path):
    model_path = _tiny_model(tmp_path)
    missing_feats = tmp_path / "no_features.jsonl"
    with TestClient(create_app(model_path, features_path=missing_feats)) as client:
        resp = client.get(
            "/profile/history",
            params={
                "zona_omi": "B12",
                "tipologia": "Abitazioni civili",
                "stato": "NORMALE",
            },
        )
        assert resp.status_code == 503


def test_meta_tipologie(tmp_path: Path):
    model_path = _tiny_model(tmp_path)
    feats = tmp_path / "features.jsonl"
    rows = [
        {"tipologia": "Negozi"},
        {"tipologia": "Abitazioni civili"},
        {"tipologia": "Abitazioni civili"},
        {"tipologia": "Ville e Villini"},
    ]
    feats.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with TestClient(create_app(model_path, features_path=feats)) as client:
        resp = client.get("/meta/tipologie")
        assert resp.status_code == 200
        assert resp.json()["tipologie"] == [
            "Abitazioni civili",
            "Negozi",
            "Ville e Villini",
        ]


def test_meta_zones(tmp_path: Path):
    model_path = _tiny_model(tmp_path)
    feats = tmp_path / "features.jsonl"
    rows = [
        {"zona_omi": "C14", "zona_omi_descr": None},
        {"zona_omi": "B12", "zona_omi_descr": "AVENTINO"},
        {"zona_omi": "B12", "zona_omi_descr": "AVENTINO"},
    ]
    feats.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with TestClient(create_app(model_path, features_path=feats)) as client:
        resp = client.get("/meta/zones")
        assert resp.status_code == 200
        zones = resp.json()["zones"]
        assert [z["zona_omi"] for z in zones] == ["B12", "C14"]
        assert zones[0]["descr"] == "AVENTINO"
        assert zones[0]["label"] == "B12 — AVENTINO"
        assert zones[1]["zona_omi"] == "C14"


def test_health_degraded_without_model(tmp_path: Path):
    missing = tmp_path / "missing.joblib"
    with TestClient(create_app(missing, features_path=tmp_path / "x.jsonl")) as client:
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
            },
        )
        assert resp.status_code == 503


def test_sighting_response(tmp_path: Path):
    model_path = _tiny_model(tmp_path)
    feats = tmp_path / "features.jsonl"
    rows = [
        {
            "zona_omi": "B12",
            "tipologia": "Abitazioni civili",
            "stato": "NORMALE",
            "semester": "2025-1",
            "price_per_m2_monthly": 10,

        }
    ]
    feats.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    
    sightings = tmp_path / "sightings.jsonl"
    with TestClient(create_app(model_path, features_path=feats, sightings_path=sightings)) as client:
        resp = client.post("/sightings", json={
            "zona_omi": "B12",
            "tipologia": "Abitazioni civili",
            "stato": "NORMALE",
            "asking_eur_m2": 10,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["zona_omi"] == "B12"
        assert body["tipologia"] == "Abitazioni civili"
        assert body["stato"] == "NORMALE"
        assert body["predicted_price_per_m2_monthly"] > 0
        assert body["asking_eur_m2"] == 10
        sightings = tmp_path / "sightings.jsonl"
        lines = sightings.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["sighting_id"] == body["sighting_id"]
    

