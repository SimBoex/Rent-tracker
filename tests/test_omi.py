"""Tests for OMI loader / features."""

from __future__ import annotations

import json
from pathlib import Path

from etl.extract.omi_loader import load_omi_csv, load_omi_dir, write_jsonl
from etl.transform.omi_features import build_features
from ml.train import train

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "omi"
FIXTURE_NAMES = (
    "quotazioni_roma_2024_1.csv",
    "quotazioni_roma_2024_2.csv",
    "quotazioni_roma_2025_1.csv",
)


def _copy_fixtures(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FIXTURE_NAMES:
        (raw_dir / name).write_text(
            (FIXTURES / name).read_text(encoding="utf-8"), encoding="utf-8"
        )


def test_load_omi_csv_filters_negozio_and_parses_decimals():
    rows = load_omi_csv(FIXTURES / "quotazioni_roma_2024_2.csv")
    assert all("negozio" not in (r.get("tipologia") or "").lower() for r in rows)
    assert len(rows) == 12
    assert rows[0]["loc_min"] == 19.0
    assert rows[0]["semester"] == "2024-2"


def test_build_features_lag_and_mid(tmp_path: Path):
    raw_dir = tmp_path / "omi"
    _copy_fixtures(raw_dir)
    rows = load_omi_dir(raw_dir)
    out = write_jsonl(rows, tmp_path / "omi.jsonl")
    assert out.is_file()
    feats = build_features(rows)
    assert all("price_per_m2_monthly" in f for f in feats)
    b12_ottimo = [
        f
        for f in feats
        if f["zona_omi"] == "B12"
        and f["stato"] == "OTTIMO"
        and f["tipologia"] == "Abitazioni civili"
    ]
    assert len(b12_ottimo) == 3
    by_sem = {f["semester"]: f for f in b12_ottimo}
    assert by_sem["2024-1"]["loc_mid_lag"] is None
    assert by_sem["2024-2"]["loc_mid_lag"] == 21.0
    assert by_sem["2024-2"]["price_per_m2_monthly"] == 22.0
    assert by_sem["2025-1"]["loc_mid_lag"] == 22.0


def test_train_omi_smoke(tmp_path: Path):
    raw_dir = tmp_path / "omi"
    _copy_fixtures(raw_dir)
    rows = load_omi_dir(raw_dir)
    feats = build_features(rows)
    feat_path = tmp_path / "features.jsonl"
    feat_path.write_text(
        "".join(json.dumps(r) + "\n" for r in feats), encoding="utf-8"
    )
    models = tmp_path / "models"
    train(input_path=feat_path, models_dir=models, tracking_uri=None)
    assert (models / "baseline_latest" / "model.joblib").is_file()
