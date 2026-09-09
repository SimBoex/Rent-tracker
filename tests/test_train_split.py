"""Unit tests for training split and smoke train (RF-05 / RNF-07)."""

from __future__ import annotations

import json
from pathlib import Path

from ml.train import DataLoader, MetricsCalculator, PipelineBuilder, Trainer, train


def _trainer(input_path: Path | None = None) -> Trainer:
    return Trainer(
        pipeline=PipelineBuilder().build(),
        metrics_calculator=MetricsCalculator(),
        data_loader=DataLoader(input_path) if input_path is not None else DataLoader(),
    )


def test_temporal_split_last_day_in_test():
    rows = [
        {"listing_id": 1, "scraped_at": "2026-09-01T10:00:00+00:00", "price_per_m2_monthly": 10},
        {"listing_id": 2, "scraped_at": "2026-09-01T11:00:00+00:00", "price_per_m2_monthly": 11},
        {"listing_id": 3, "scraped_at": "2026-09-08T10:00:00+00:00", "price_per_m2_monthly": 12},
        {"listing_id": 4, "scraped_at": "2026-09-08T11:00:00+00:00", "price_per_m2_monthly": 13},
    ]
    train_rows, test_rows, mode = _trainer().temporal_or_ordered_split(rows)
    assert mode == "temporal_last_day"
    assert {r["listing_id"] for r in train_rows} == {1, 2}
    assert {r["listing_id"] for r in test_rows} == {3, 4}


def test_single_day_ordered_holdout_fallback():
    rows = [
        {
            "listing_id": i,
            "scraped_at": "2026-09-08T10:00:00+00:00",
            "price_per_m2_monthly": float(i),
        }
        for i in range(1, 11)
    ]
    train_rows, test_rows, mode = _trainer().temporal_or_ordered_split(rows, holdout_frac=0.2)
    assert mode == "ordered_holdout_fallback"
    assert len(train_rows) == 8
    assert len(test_rows) == 2
    assert [r["listing_id"] for r in test_rows] == [9, 10]


def test_load_drops_missing_municipio(tmp_path: Path):
    rows = [
        {"listing_id": 1, "price_per_m2_monthly": 20.0, "municipio": "I"},
        {"listing_id": 2, "price_per_m2_monthly": 21.0, "municipio": None},
        {"listing_id": 3, "price_per_m2_monthly": 22.0, "municipio": ""},
        {"listing_id": 4, "price_per_m2_monthly": None, "municipio": "II"},
    ]
    path = tmp_path / "features.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    loaded = DataLoader(path).load()
    assert len(loaded) == 1
    assert loaded[0]["listing_id"] == 1


def test_train_smoke(tmp_path: Path):
    rows = []
    for i in range(20):
        rows.append(
            {
                "listing_id": i,
                "scraped_at": "2026-09-08T10:00:00+00:00",
                "price_per_m2_monthly": 20.0 + i * 0.5,
                "surface_m2": 50.0 + i,
                "rooms": 2,
                "distance_from_center_km": 3.0,
                "area_price_per_m2_hist": 22.0,
                "publication_month": 9,
                "municipio": "I" if i % 2 == 0 else "II",
            }
        )
    # one row without municipio must be dropped before split/train
    rows.append(
        {
            "listing_id": 99,
            "scraped_at": "2026-09-08T10:00:00+00:00",
            "price_per_m2_monthly": 25.0,
            "surface_m2": 60.0,
            "rooms": 2,
            "distance_from_center_km": 3.0,
            "area_price_per_m2_hist": 22.0,
            "publication_month": 9,
            "municipio": None,
        }
    )
    inp = tmp_path / "features.jsonl"
    inp.write_text(
        "".join(json.dumps(r) + "\n" for r in rows),
        encoding="utf-8",
    )
    models_dir = tmp_path / "models"
    out = train(input_path=inp, models_dir=models_dir, tracking_uri=None)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert (out / "model.joblib").is_file()
    assert metrics["split_mode"] == "ordered_holdout_fallback"
    assert metrics["n_train"] + metrics["n_test"] == 20
    assert "mae" in metrics and "rmse" in metrics and "r2" in metrics
    assert (models_dir / "baseline_latest" / "metrics.json").is_file()
