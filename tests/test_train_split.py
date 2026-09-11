"""Unit tests for training split and smoke train (RF-05 / RNF-07)."""

from __future__ import annotations

import json
from pathlib import Path

from ml.train import DataLoader, MetricsCalculator, PipelineBuilder, Trainer, train
from tests.omi_rows import omi_feature_row


def _trainer(input_path: Path | None = None) -> Trainer:
    return Trainer(
        pipeline=PipelineBuilder().build(),
        metrics_calculator=MetricsCalculator(),
        data_loader=DataLoader(input_path) if input_path is not None else DataLoader(),
    )


def test_temporal_split_last_day_in_test():
    rows = [
        omi_feature_row(1, day="2026-09-01"),
        omi_feature_row(2, day="2026-09-01"),
        omi_feature_row(3, day="2026-09-08"),
        omi_feature_row(4, day="2026-09-08"),
    ]
    train_rows, test_rows, mode = _trainer().temporal_or_ordered_split(rows)
    assert mode == "temporal_last_day"
    assert {r["listing_id"] for r in train_rows} == {rows[0]["listing_id"], rows[1]["listing_id"]}
    assert {r["listing_id"] for r in test_rows} == {rows[2]["listing_id"], rows[3]["listing_id"]}


def test_single_day_ordered_holdout_fallback():
    rows = [omi_feature_row(i, day="2026-09-08", loc_mid_lag=10.0 + i) for i in range(1, 11)]
    train_rows, test_rows, mode = _trainer().temporal_or_ordered_split(rows, holdout_frac=0.2)
    assert mode == "ordered_holdout_fallback"
    assert len(train_rows) == 8
    assert len(test_rows) == 2


def test_load_drops_missing_zona(tmp_path: Path):
    rows = [
        {**omi_feature_row(1), "zona_omi": "B12"},
        {**omi_feature_row(2), "zona_omi": None},
        {**omi_feature_row(3), "zona_omi": ""},
        {**omi_feature_row(4), "price_per_m2_monthly": None},
    ]
    path = tmp_path / "features.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    loaded = DataLoader(path).load()
    assert len(loaded) == 1
    assert loaded[0]["zona_omi"] == "B12"


def test_train_smoke(tmp_path: Path):
    rows = [
        omi_feature_row(i, day="2026-09-01" if i < 12 else "2026-09-08", loc_mid_lag=12.0 + i)
        for i in range(24)
    ]
    rows.append({**omi_feature_row(99, day="2026-09-08"), "zona_omi": None})
    inp = tmp_path / "features.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    models_dir = tmp_path / "models"
    out = train(input_path=inp, models_dir=models_dir, tracking_uri=None)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert (out / "model.joblib").is_file()
    assert metrics["split_mode"] == "temporal_last_day"
    assert metrics["n_train"] + metrics["n_test"] == 24
    assert "mae" in metrics and "rmse" in metrics and "r2" in metrics
    assert (models_dir / "baseline_latest" / "metrics.json").is_file()
