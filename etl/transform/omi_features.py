"""Transform OMI quotazioni JSONL → features for training (no loc min/max leakage)."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "raw" / "omi" / "omi_quotazioni_latest.jsonl"
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_OUT = PROCESSED_DIR / "features_latest.jsonl"

TARGET = "price_per_m2_monthly"
logger = logging.getLogger(__name__)


def _semester_key(sem: str) -> tuple[int, int]:
    """Sort key for 'YYYY-S' or loose strings."""
    parts = str(sem).replace("_", "-").split("-")
    try:
        year = int(parts[0])
        half = int(parts[1]) if len(parts) > 1 else 1
        return year, half
    except ValueError:
        return (0, 0)


def build_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mid target + lag mid from previous semester for same zona/tipologia/stato."""
    by_key: dict[tuple[Any, ...], list[tuple[tuple[int, int], float]]] = {}
    for row in rows:
        mid = (float(row["loc_min"]) + float(row["loc_max"])) / 2.0
        key = (row.get("zona_omi"), row.get("tipologia"), row.get("stato"))
        by_key.setdefault(key, []).append((_semester_key(row["semester"]), mid))
    for series in by_key.values():
        series.sort(key=lambda x: x[0])

    enriched: list[dict[str, Any]] = []
    for row in rows:
        mid = (float(row["loc_min"]) + float(row["loc_max"])) / 2.0
        key = (row.get("zona_omi"), row.get("tipologia"), row.get("stato"))
        year, half = _semester_key(row["semester"])
        month = 6 if half == 1 else 12
        scraped_at = f"{year}-{month:02d}-15T00:00:00+00:00"
        lag = None
        series = by_key.get(key, [])
        sk = _semester_key(row["semester"])
        for i, (sem_k, _) in enumerate(series):
            if sem_k == sk:
                if i > 0:
                    lag = series[i - 1][1]
                break
        enriched.append(
            {
                "source": row.get("source", "agenziaentrate-omi"),
                "listing_id": (
                    f"{row.get('zona_omi')}|{row.get('tipologia')}|"
                    f"{row.get('stato')}|{row.get('semester')}"
                ),
                "zona_omi": row.get("zona_omi"),
                "zona_omi_descr": row.get("zona_omi_descr"),
                "tipologia": row.get("tipologia"),
                "stato": row.get("stato"),
                "semester": row.get("semester"),
                "loc_mid_lag": lag,
                "omi_loc_min": float(row["loc_min"]),
                "omi_loc_max": float(row["loc_max"]),
                TARGET: mid,
                "scraped_at": scraped_at,
                "publication_month": month,
            }
        )
    return enriched


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_features(rows: list[dict[str, Any]], out_path: Path = DEFAULT_OUT) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamped = (
        PROCESSED_DIR
        / f"features_omi_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    )
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    out_path.write_text(payload, encoding="utf-8")
    stamped.write_text(payload, encoding="utf-8")
    logger.info("Wrote %s features → %s", len(rows), out_path)
    return out_path


def run(input_path: Path = DEFAULT_INPUT, out_path: Path = DEFAULT_OUT) -> Path:
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing {input_path}; run python -m etl.extract.omi_loader first")
    rows = load_jsonl(input_path)
    features = build_features(rows)
    return write_features(features, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="OMI JSONL → features_latest.jsonl.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(args.input, args.out)


if __name__ == "__main__":
    main()
