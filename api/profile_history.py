"""Profile semester history from processed OMI features (zona / tipologia / stato)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from etl.semester import semester_key
from ml.train import TARGET


def load_profile_series(
    features_path: Path,
    *,
    zona_omi: str,
    tipologia: str,
    stato: str,
) -> list[dict[str, Any]]:
    """Return mid OMI points for one profile, oldest → newest. Empty if none/missing file."""
    if not features_path.is_file():
        return []

    zona = zona_omi.strip()
    matched: list[dict[str, Any]] = []
    for line in features_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("zona_omi") != zona:
            continue
        if row.get("tipologia") != tipologia:
            continue
        if row.get("stato") != stato:
            continue
        mid = row.get(TARGET)
        semester = row.get("semester")
        if mid is None or semester is None or semester == "":
            continue
        matched.append(
            {
                "semester": str(semester),
                "price_per_m2_monthly": float(mid),
                "omi_loc_min": (
                    None if row.get("omi_loc_min") is None else float(row["omi_loc_min"])
                ),
                "omi_loc_max": (
                    None if row.get("omi_loc_max") is None else float(row["omi_loc_max"])
                ),
                "loc_mid_lag": (
                    None if row.get("loc_mid_lag") is None else float(row["loc_mid_lag"])
                ),
            }
        )

    matched.sort(key=lambda r: semester_key(r["semester"]))
    if not matched:
        return []

    last = matched[-1]["semester"]
    for point in matched:
        point["role"] = "test" if point["semester"] == last else "train"
    return matched
