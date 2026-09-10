"""Keep a rolling window of raw scrapes (and drop HTML / old feature stamps)."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

logger = logging.getLogger(__name__)


def prune_raw(raw_dir: Path = RAW_DIR, keep: int = 1) -> list[Path]:
    """Delete older immobiliare_roma_* dirs; keep the newest ``keep`` ones."""
    keep = max(0, keep)
    dirs = sorted(p for p in raw_dir.glob("immobiliare_roma_*") if p.is_dir())
    drop = dirs if keep == 0 else dirs[:-keep]
    for path in drop:
        shutil.rmtree(path)
        logger.info("Removed old scrape %s", path.name)
    # Drop any leftover HTML in kept dirs (CI uses --no-html; local may still have dumps).
    for html in raw_dir.glob("**/*.html"):
        html.unlink()
        logger.info("Removed HTML %s", html.relative_to(raw_dir))
    kept = sorted(p for p in raw_dir.glob("immobiliare_roma_*") if p.is_dir())
    logger.info("Raw scrapes kept: %s", [p.name for p in kept])
    return kept


def prune_feature_stamps(processed_dir: Path = PROCESSED_DIR) -> int:
    """Delete timestamped features_*.jsonl; keep features_latest.jsonl."""
    n = 0
    for path in processed_dir.glob("features_*.jsonl"):
        if path.name == "features_latest.jsonl":
            continue
        path.unlink()
        n += 1
        logger.info("Removed feature stamp %s", path.name)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune data/raw to the newest N scrapes; drop HTML and old feature stamps."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument(
        "--keep",
        type=int,
        default=1,
        help="Newest immobiliare_roma_* dirs to keep (default 1)",
    )
    parser.add_argument(
        "--no-feature-stamps",
        action="store_true",
        help="Do not delete timestamped features_*.jsonl",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    prune_raw(raw_dir=args.raw_dir, keep=args.keep)
    if not args.no_feature_stamps:
        prune_feature_stamps(processed_dir=args.processed_dir)


if __name__ == "__main__":
    main()
