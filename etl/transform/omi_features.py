"""Transform OMI quotazioni JSONL → features for training (no loc min/max leakage)."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from etl.semester import semester_key
from etl.jsonl import load_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "raw" / "omi" / "omi_quotazioni_latest.jsonl"
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_OUT = PROCESSED_DIR / "features_latest.jsonl"

TARGET = "price_per_m2_monthly"
logger = logging.getLogger(__name__)


def build_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mid target + lag mid from previous semester for same zona/tipologia/stato."""
    # Build a dict of lists of tuples (semester key, mid) by zona/tipologia/stato
    by_key: dict[tuple[Any, ...], list[tuple[tuple[int, int], float]]] = {}

    for row in rows:
        mid = (float(row["loc_min"]) + float(row["loc_max"])) / 2.0
        key = (row.get("zona_omi"), row.get("tipologia"), row.get("stato"))
        by_key.setdefault(key, []).append((semester_key(row["semester"]), mid))

    # ordering by year and half
    for series in by_key.values():
        series.sort(key=lambda x: x[0])

    enriched: list[dict[str, Any]] = []

    for row in rows:
        mid = (float(row["loc_min"]) + float(row["loc_max"])) / 2.0
        key = (row.get("zona_omi"), row.get("tipologia"), row.get("stato"))
        sk = semester_key(row["semester"])

        # mid location price at previous semester
        lag = None
        series = by_key.get(key, [])

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
                # mid location price at previous semester
                "loc_mid_lag": lag,
                # min and max location price
                "omi_loc_min": float(row["loc_min"]),
                "omi_loc_max": float(row["loc_max"]),
                TARGET: mid,
            }
        )
    return enriched


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
