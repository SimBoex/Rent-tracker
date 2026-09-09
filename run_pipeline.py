#!/usr/bin/env python3
"""Run the full pipeline: optional scrape → clean → features → train."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roma Rent Monitor pipeline: scrape (opt) → clean → features → train."
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Reuse existing data/raw (default: scrape first)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Stop after features (no training)",
    )
    parser.add_argument("--start-page", type=int, default=1, help="First scrape page (default 1)")
    parser.add_argument("--max-pages", type=int, default=1, help="Pages to fetch from start-page")
    parser.add_argument("--no-html", action="store_true", help="Do not save raw HTML pages")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging in train")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    v = ["-v"] if args.verbose else []

    if not args.skip_scrape:
        scrape = [
            py,
            "etl/extract/immobiliare_scraper.py",
            "--start-page",
            str(args.start_page),
            "--max-pages",
            str(args.max_pages),
            *v,
        ]
        if args.no_html:
            scrape.append("--no-html")
        _run(scrape)

    _run([py, "-m", "etl.transform.clean_phase", *v])
    _run([py, "-m", "etl.transform.features_phase", *v])

    if not args.skip_train:
        train = [py, "-m", "ml.train", *v]
        if args.no_mlflow:
            train.append("--no-mlflow")
        _run(train)

    print("Pipeline done.", flush=True)


if __name__ == "__main__":
    main()
