"""Distinct OMI zones (code + description) from features JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from etl.extract.omi_loader import RAW_OMI_DIR, load_omi_zone_catalog_dir

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = _ROOT / "data" / "processed" / "features_latest.jsonl"


def zone_label(zona_omi: str, descr: str | None) -> str:
    """UI label: ``B12 — AVENTINO…`` or just the code."""
    code = zona_omi.strip()
    if descr and str(descr).strip():
        return f"{code} — {str(descr).strip()}"
    return code


def list_zones(
    path: Path = DEFAULT_FEATURES,
    *,
    raw_omi_dir: Path | None = RAW_OMI_DIR,
) -> list[dict[str, Any]]:
    """Distinct zones from features JSONL, with optional ZONE.csv fallback for descr.

    Returns sorted list of ``{zona_omi, descr, label}``. Empty if file missing.
    """
    by_code: dict[str, str | None] = {}
    if path.is_file():
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    zona = row.get("zona_omi")
                    if not isinstance(zona, str) or not zona.strip():
                        continue
                    code = zona.strip()
                    descr = row.get("zona_omi_descr")
                    if isinstance(descr, str) and descr.strip():
                        by_code[code] = descr.strip()
                    else:
                        by_code.setdefault(code, None)
        except (OSError, json.JSONDecodeError):
            pass

    catalog: dict[str, str] = {}
    if raw_omi_dir is not None and raw_omi_dir.is_dir():
        try:
            catalog = load_omi_zone_catalog_dir(raw_omi_dir)
        except OSError:
            catalog = {}

    out: list[dict[str, Any]] = []
    for code in sorted(by_code):
        descr = by_code[code] or catalog.get(code)
        out.append(
            {
                "zona_omi": code,
                "descr": descr,
                "label": zone_label(code, descr),
            }
        )
    return out
