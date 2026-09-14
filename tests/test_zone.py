"""Tests for api.zone meta helpers."""

from __future__ import annotations

import json
from pathlib import Path

from api.zone import list_zones, zone_label


def test_zone_label():
    assert zone_label("B12", None) == "B12"
    assert zone_label("B12", "  AVENTINO  ") == "B12 — AVENTINO"


def test_list_zones_from_features_and_catalog(tmp_path: Path):
    feats = tmp_path / "features.jsonl"
    rows = [
        {"zona_omi": "C14", "zona_omi_descr": None},
        {"zona_omi": "B12", "zona_omi_descr": "AVENTINO"},
    ]
    feats.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    empty_raw = tmp_path / "omi"
    empty_raw.mkdir()
    out = list_zones(feats, raw_omi_dir=empty_raw)
    assert out == [
        {"zona_omi": "B12", "descr": "AVENTINO", "label": "B12 — AVENTINO"},
        {"zona_omi": "C14", "descr": None, "label": "C14"},
    ]

    zone_csv = empty_raw / "QI_x_ZONE.csv"
    zone_csv.write_text(
        "Comune_descrizione;Zona_Descr;Zona;LinkZona\n"
        "ROMA;'EUR centro';C14;RM2\n",
        encoding="utf-8",
    )
    out2 = list_zones(feats, raw_omi_dir=empty_raw)
    assert out2[1]["descr"] == "EUR centro"
    assert out2[1]["label"] == "C14 — EUR centro"
