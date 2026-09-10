"""Fingerprint training datasets for RF-12 (content hash, no public data push)."""

from __future__ import annotations

from typing import Any
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json


def fingerprint(path: Path) -> dict[str, Any]:
    """Return a content-addressed identity for a features JSONL (or any file)."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset not found: {resolved}")

    h = hashlib.sha256()
    n_rows = 0
    with resolved.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    # Count non-empty lines (JSONL rows) without a second full decode pass if small —
    # second pass in text is fine for portfolio sizes.
    with resolved.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n_rows += 1

    stat = resolved.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return {
        "path": str(resolved),
        "sha256": h.hexdigest(),
        "nbytes": int(stat.st_size),
        "n_rows": n_rows,
        "mtime_utc": mtime.isoformat(),
    }


def write_dataset_json(meta: dict[str, Any], out_path: Path) -> None:
    out_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
