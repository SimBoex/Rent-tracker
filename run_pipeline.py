#!/usr/bin/env python3
"""OMI pipeline: CSV → features → train (manual download into data/raw/omi/)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "tests" / "fixtures" / "omi"
RAW_OMI = ROOT / "data" / "raw" / "omi"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OMI: load CSV → features → train. Put CSVs in data/raw/omi/ (see doc/omi.md)."
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="Copy tests/fixtures/omi/*.csv into data/raw/omi/ before load",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    v = ["-v"] if args.verbose else []

    if args.use_fixture:
        RAW_OMI.mkdir(parents=True, exist_ok=True)
        for src in FIXTURES.glob("*.csv"):
            shutil.copy2(src, RAW_OMI / src.name)
            print(f"Copied fixture {src.name}", flush=True)

    _run([py, "-m", "etl.extract.omi_loader", *v])
    _run([py, "-m", "etl.transform.omi_features", *v])
    if not args.skip_train:
        train = [py, "-m", "ml.train", *v]
        if args.no_mlflow:
            train.append("--no-mlflow")
        _run(train)
    print("OMI pipeline done.", flush=True)


if __name__ == "__main__":
    main()
