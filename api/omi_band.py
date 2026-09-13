"""Lookup OMI locazione min/max band from processed features (not model uncertainty)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from etl.semester import semester_key


def band_payload(loc_min: float, loc_max: float) -> dict[str, Any]:
    lo = float(loc_min)
    hi = float(loc_max)
    return {
        "omi_loc_min": round(lo, 4),
        "omi_loc_max": round(hi, 4),
        "omi_half_width": round((hi - lo) / 2.0, 4),
        "band_source": "omi",
    }


def build_band_index(features_path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Map (zona_omi, tipologia, stato) → band from the latest OMI semester in JSONL."""
    if not features_path.is_file():
        return {}

    best: dict[tuple[Any, ...], tuple[tuple[int, int], dict[str, Any]]] = {}
    for line in features_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        lo, hi = row.get("omi_loc_min"), row.get("omi_loc_max")
        if lo is None or hi is None:
            continue
        key = (row.get("zona_omi"), row.get("tipologia"), row.get("stato"))
        if key[0] is None or key[0] == "":
            continue
        sem = row.get("semester") or ""
        sk = semester_key(str(sem))
        prev = best.get(key)
        if prev is None or sk >= prev[0]:
            best[key] = (sk, band_payload(float(lo), float(hi)))

    return {k: payload for k, (_sk, payload) in best.items()}
