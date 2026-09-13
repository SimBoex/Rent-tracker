"""CI helper: pull OMI CSVs uploaded via cloud ingest inbox into data/raw/omi/."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from api.cloud_store import cloud_configured, download_inbox
from etl.extract.omi_loader import RAW_OMI_DIR

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download omi-ingest inbox from R2/S3 into data/raw/omi/"
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_OMI_DIR)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not cloud_configured():
        logger.error("Missing AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
        raise SystemExit(1)
    written = download_inbox(args.raw_dir)
    logger.info("Pulled %s file(s) from ingest inbox", len(written))


if __name__ == "__main__":
    main()
