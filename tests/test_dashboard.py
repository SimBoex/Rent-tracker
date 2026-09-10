"""Unit tests for dashboard data helpers (RF-10)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from api.predictor import GOOD_DEAL, ModelPredictor
from dashboard.data import (
    good_deals_table,
    load_feature_rows,
    load_json,
    score_listings,
)
from ml.train import train


def test_load_json_missing(tmp_path: Path):
    assert load_json(tmp_path / "missing.json") is None


def test_load_json_ok(tmp_path: Path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"a": 1}) + "\n", encoding="utf-8")
    assert load_json(path) == {"a": 1}


def test_score_and_good_deals(tmp_path: Path):
    features = tmp_path / "features.jsonl"
    rows = []
    for i in range(20):
        day = "2026-09-01" if i < 10 else "2026-09-08"
        # Low actual price on even ids → likely good_deal vs model
        price = 10.0 if i % 2 == 0 else 40.0 + i
        rows.append(
            {
                "listing_id": i,
                "url": f"https://example.com/{i}",
                "scraped_at": f"{day}T10:00:00+00:00",
                "price_per_m2_monthly": price,
                "surface_m2": 50.0 + i,
                "rooms": 2,
                "distance_from_center_km": 3.0,
                "area_price_per_m2_hist": 22.0,
                "publication_month": 9,
                "municipio": "I" if i % 2 == 0 else "II",
            }
        )
    features.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    models_dir = tmp_path / "models"
    train(input_path=features, models_dir=models_dir, tracking_uri=None)
    model_path = models_dir / "baseline_latest" / "model.joblib"
    assert model_path.is_file()

    loaded = load_feature_rows(features, limit=20)
    assert len(loaded) == 20
    predictor = ModelPredictor(model_path)
    scored = score_listings(loaded, predictor)
    assert len(scored) == 20
    assert "deal_label" in scored.columns
    deals = good_deals_table(scored)
    assert (deals["deal_label"] == GOOD_DEAL).all() if not deals.empty else True
    assert list(deals["gap_pct"]) == sorted(deals["gap_pct"].tolist()) if not deals.empty else True
    # With deliberately cheap listings, expect at least one good deal
    assert not deals.empty
    assert joblib.load(model_path) is not None
