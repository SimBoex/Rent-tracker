"""Unit tests for Evidently drift report skeleton (RF-08)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from ml.drift_report import DriftFrameBuilder, DriftReporter, generate_drift_report
from ml.train import FEATURE_COLS, TARGET, DataLoader, train
from tests.omi_rows import omi_feature_row


def _write_features(path: Path, n: int = 24, two_days: bool = True) -> None:
    rows = []
    for i in range(n):
        day = "2026-09-01" if two_days and i < n // 2 else "2026-09-08"
        rows.append(omi_feature_row(i, day=day, loc_mid_lag=12.0 + i))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_frame_builder_temporal_split_and_prediction(tmp_path: Path):
    features = tmp_path / "features.jsonl"
    _write_features(features, n=24, two_days=True)
    models_dir = tmp_path / "models"
    train(input_path=features, models_dir=models_dir, tracking_uri=None)
    model_path = models_dir / "baseline_latest" / "model.joblib"
    assert model_path.is_file()

    builder = DriftFrameBuilder(DataLoader(features), model_path=model_path)
    ref, cur = builder.build()
    assert builder.split_mode == "temporal_last_day"
    assert len(ref) == 12 and len(cur) == 12
    assert list(ref.columns) == FEATURE_COLS + [TARGET, "prediction"]
    assert "prediction" in cur.columns
    assert joblib.load(model_path) is not None


def test_generate_drift_report_smoke(tmp_path: Path):
    features = tmp_path / "features.jsonl"
    _write_features(features, n=24, two_days=True)
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
    assert summary["n_reference"] == 12
    assert summary["n_current"] == 12
    assert summary["mae_by_day"] is not None
    assert len(summary["mae_by_day"]) == 2
    assert {d["day"] for d in summary["mae_by_day"]} == {"2026-09-01", "2026-09-08"}
    assert summary["mae_reference"] is not None
    assert summary["mae_current"] is not None
    assert (reports_dir / "drift_latest" / "summary.json").is_file()


def test_drift_without_model(tmp_path: Path):
    features = tmp_path / "features.jsonl"
    _write_features(features, n=12, two_days=True)
    builder = DriftFrameBuilder(DataLoader(features), model_path=tmp_path / "missing.joblib")
    ref, cur = builder.build()
    assert "prediction" not in ref.columns
    assert TARGET in cur.columns
    out = DriftReporter(builder).run(reports_dir=tmp_path / "reports")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["has_prediction"] is False
    assert summary["mae_by_day"] is None
