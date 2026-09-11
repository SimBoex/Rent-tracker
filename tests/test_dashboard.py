"""Unit tests for dashboard data helpers (RF-10)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from api.predictor import BELOW_OMI_BAND, ModelPredictor
from dashboard.data import (
    export_good_deals,
    export_monitoring_snapshot,
    good_deals_table,
    load_feature_rows,
    load_good_deals_snapshot,
    load_json,
    load_monitoring_snapshot,
    score_rows,
)
from ml.train import train
from tests.omi_rows import omi_feature_row


def test_load_json_missing(tmp_path: Path):
    assert load_json(tmp_path / "missing.json") is None


def test_load_json_ok(tmp_path: Path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"a": 1}) + "\n", encoding="utf-8")
    assert load_json(path) == {"a": 1}


def test_score_and_good_deals(tmp_path: Path):
    features = tmp_path / "features.jsonl"
    rows = []
    for i in range(24):
        day = "2026-09-01" if i < 12 else "2026-09-08"
        price = 8.0 if i % 2 == 0 else 40.0 + i
        rows.append(omi_feature_row(i, day=day, price=price, loc_mid_lag=15.0 + i))
    features.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    models_dir = tmp_path / "models"
    train(input_path=features, models_dir=models_dir, tracking_uri=None)
    model_path = models_dir / "baseline_latest" / "model.joblib"
    assert model_path.is_file()

    loaded = load_feature_rows(features, limit=24)
    assert len(loaded) == 24
    predictor = ModelPredictor(model_path)
    scored = score_rows(loaded, predictor)
    assert len(scored) == 24
    assert "deal_label" in scored.columns
    deals = good_deals_table(scored)
    assert (deals["deal_label"] == BELOW_OMI_BAND).all() if not deals.empty else True
    assert list(deals["gap_pct"]) == sorted(deals["gap_pct"].tolist()) if not deals.empty else True
    assert not deals.empty
    assert joblib.load(model_path) is not None

    out = tmp_path / "good_deals_latest.json"
    export_good_deals(features_path=features, model_path=model_path, out_path=out, limit=24)
    snap = load_good_deals_snapshot(path=out, url="")
    assert snap is not None
    assert snap["n_good_deals"] >= 1
    assert snap["rows"]
    assert snap.get("source_attribution") == "Agenzia Entrate – OMI"
    row0 = snap["rows"][0]
    assert "url" not in row0
    assert "listing_id" not in row0
    assert "price_per_m2_monthly" not in row0
    assert "gap_pct" not in row0
    assert "predicted_price_per_m2_monthly" not in row0
    assert "deal_label" in row0
    assert load_good_deals_snapshot(path=tmp_path / "missing.json", url="") is None

    drift_path = tmp_path / "summary.json"
    drift_path.write_text(
        json.dumps(
            {
                "input": "/secret/path",
                "n_reference": 10,
                "n_current": 5,
                "drifted_columns_share": 0.1,
                "mae_reference": {"mae": 1.0},
                "mae_current": {"mae": 1.2},
            }
        ),
        encoding="utf-8",
    )
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(
        json.dumps({"should_retrain": False, "trigger_reason": "ok", "mae_ratio": 1.2}),
        encoding="utf-8",
    )
    mon_out = tmp_path / "monitoring_latest.json"
    export_monitoring_snapshot(drift_path=drift_path, decision_path=decision_path, out_path=mon_out)
    mon = load_monitoring_snapshot(path=mon_out, url="")
    assert mon is not None
    assert mon["drift"]["n_reference"] == 10
    assert "input" not in mon["drift"]
    assert mon["decision"]["should_retrain"] is False
