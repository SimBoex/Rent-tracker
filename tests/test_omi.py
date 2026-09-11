"""Tests for OMI loader / features (inline mini-CSV — no fixtures dir)."""

from __future__ import annotations

import json
from pathlib import Path

from etl.extract.omi_loader import load_omi_csv, load_omi_dir, write_jsonl
from etl.transform.omi_features import build_features
from ml.train import train

# Minimal official-shaped snippets (synthetic; not real OMI dumps).
_CSV_2024_1 = """\
COMUNE_DESCRIZIONE;ZONA;ZONA_DESCR;DESCR_TIPOLOGIA;STATO;LOCMIN;LOCMAX
Roma;B12;Centro;Abitazioni civili;OTTIMO;18,0;24,0
Roma;B12;Centro;Abitazioni civili;NORMALE;14,0;18,0
Roma;C14;Eur;Abitazioni civili;NORMALE;12,0;15,0
Roma;C14;Eur;Abitazioni civili;OTTIMO;14,0;17,0
Roma;D20;Periferia;Abitazioni civili;NORMALE;8,0;11,0
Roma;E5;Nord;Abitazioni civili;NORMALE;10,0;13,0
Roma;F2;Est;Abitazioni civili;NORMALE;9,0;12,0
Roma;G3;Ovest;Abitazioni civili;NORMALE;11,0;14,0
"""

_CSV_2024_2 = """\
COMUNE_DESCRIZIONE;ZONA;ZONA_DESCR;DESCR_TIPOLOGIA;STATO;LOCMIN;LOCMAX
Roma;B12;Centro;Abitazioni civili;OTTIMO;19,0;25,0
Roma;B12;Centro;Abitazioni civili;NORMALE;14,5;18,5
Roma;C14;Eur;Abitazioni civili;NORMALE;12,5;15,5
Roma;C14;Eur;Abitazioni civili;OTTIMO;14,5;17,5
Roma;D20;Periferia;Abitazioni civili;NORMALE;8,5;11,5
Roma;E5;Nord;Abitazioni civili;NORMALE;10,5;13,5
Roma;F2;Est;Abitazioni civili;NORMALE;9,5;12,5
Roma;G3;Ovest;Abitazioni civili;NORMALE;11,5;14,5
Roma;E1;X;Negozio;NORMALE;30,0;40,0
"""

_CSV_2025_1 = """\
COMUNE_DESCRIZIONE;ZONA;ZONA_DESCR;DESCR_TIPOLOGIA;STATO;LOCMIN;LOCMAX
Roma;B12;Centro;Abitazioni civili;OTTIMO;20,0;26,0
Roma;B12;Centro;Abitazioni civili;NORMALE;15,0;19,0
Roma;C14;Eur;Abitazioni civili;NORMALE;13,0;16,0
Roma;C14;Eur;Abitazioni civili;OTTIMO;15,0;18,0
Roma;D20;Periferia;Abitazioni civili;NORMALE;9,0;12,0
Roma;E5;Nord;Abitazioni civili;NORMALE;11,0;14,0
Roma;F2;Est;Abitazioni civili;NORMALE;10,0;13,0
Roma;G3;Ovest;Abitazioni civili;NORMALE;12,0;15,0
"""


def _write_semester_csvs(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "quotazioni_roma_2024_1.csv").write_text(_CSV_2024_1, encoding="utf-8")
    (raw_dir / "quotazioni_roma_2024_2.csv").write_text(_CSV_2024_2, encoding="utf-8")
    (raw_dir / "quotazioni_roma_2025_1.csv").write_text(_CSV_2025_1, encoding="utf-8")


def test_load_omi_csv_filters_negozio_and_parses_decimals(tmp_path: Path):
    path = tmp_path / "quotazioni_roma_2024_2.csv"
    path.write_text(_CSV_2024_2, encoding="utf-8")
    rows = load_omi_csv(path)
    assert all("negozio" not in (r.get("tipologia") or "").lower() for r in rows)
    assert len(rows) == 8
    assert rows[0]["loc_min"] == 19.0
    assert rows[0]["semester"] == "2024-2"


def test_build_features_lag_and_mid(tmp_path: Path):
    raw_dir = tmp_path / "omi"
    _write_semester_csvs(raw_dir)
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
    _write_semester_csvs(raw_dir)
    rows = load_omi_dir(raw_dir)
    feats = build_features(rows)
    feat_path = tmp_path / "features.jsonl"
    feat_path.write_text(
        "".join(json.dumps(r) + "\n" for r in feats), encoding="utf-8"
    )
    models = tmp_path / "models"
    train(input_path=feat_path, models_dir=models, tracking_uri=None)
    assert (models / "baseline_latest" / "model.joblib").is_file()
