"""Unit tests for Evidently drift report skeleton (RF-08)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from ml.drift_report import DriftFrameBuilder, DriftReporter, generate_drift_report
from ml.train import DataLoader, FEATURE_COLS, TARGET, train


def _write_features(path: Path, n: int = 20, two_days: bool = True) -> None:
    rows = []
    for i in range(n):
        day = "2026-09-01" if two_days and i < n // 2 else "2026-09-08"
        rows.append(
            {
                "listing_id": i,
                "scraped_at": f"{day}T10:00:00+00:00",
                "price_per_m2_monthly": 20.0 + i * 0.5,
                "surface_m2": 50.0 + i,
                "rooms": 2,
                "distance_from_center_km": 3.0,
                "area_price_per_m2_hist": 22.0,
                "publication_month": 9,
                "municipio": "I" if i % 2 == 0 else "II",
            }
        )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_frame_builder_temporal_split_and_prediction(tmp_path: Path):
    features = tmp_path / "features.jsonl"
    _write_features(features, n=20, two_days=True)
    models_dir = tmp_path / "models"
    train(input_path=features, models_dir=models_dir, tracking_uri=None)
    model_path = models_dir / "baseline_latest" / "model.joblib"
    assert model_path.is_file()

    builder = DriftFrameBuilder(DataLoader(features), model_path=model_path)
    ref, cur = builder.build()
    assert builder.split_mode == "temporal_last_day"
    assert len(ref) == 10 and len(cur) == 10
    assert list(ref.columns) == FEATURE_COLS + [TARGET, "prediction"]
    assert "prediction" in cur.columns
    assert joblib.load(model_path) is not None


def test_generate_drift_report_smoke(tmp_path: Path):
    features = tmp_path / "features.jsonl"
    _write_features(features, n=20, two_days=True)
    models_dir = tmp_path / "models"
    train(input_path=features, models_dir=models_dir, tracking_uri=None)

    reports_dir = tmp_path / "reports"
    out = generate_drift_report(
        input_path=features,
        model_path=models_dir / "baseline_latest" / "model.joblib",
        reports_dir=reports_dir,
    )
    assert (out / "report.html").is_file()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["split_mode"] == "temporal_last_day"
    assert summary["has_prediction"] is True
    assert summary["n_reference"] == 10
    assert summary["n_current"] == 10
    assert summary["mae_by_day"] is not None
    assert len(summary["mae_by_day"]) == 2
    assert {d["day"] for d in summary["mae_by_day"]} == {"2026-09-01", "2026-09-08"}
    for day in summary["mae_by_day"]:
        assert day["n"] == 10
        assert isinstance(day["mae"], float)
        assert isinstance(day["rmse"], float)
    assert summary["mae_reference"] is not None
    assert summary["mae_current"] is not None
    assert summary["mae_reference"]["n"] == 10
    assert summary["mae_current"]["n"] == 10
    assert isinstance(summary["mae_reference"]["mae"], float)
    assert isinstance(summary["mae_current"]["mae"], float)
    assert (reports_dir / "drift_latest" / "summary.json").is_file()


def test_drift_without_model(tmp_path: Path):
    features = tmp_path / "features.jsonl"
    _write_features(features, n=10, two_days=True)
    builder = DriftFrameBuilder(DataLoader(features), model_path=tmp_path / "missing.joblib")
    ref, cur = builder.build()
    assert "prediction" not in ref.columns
    assert TARGET in cur.columns
    out = DriftReporter(builder).run(reports_dir=tmp_path / "reports")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["has_prediction"] is False
    assert summary["mae_by_day"] is None
    assert summary["mae_reference"] is None
    assert summary["mae_current"] is None
