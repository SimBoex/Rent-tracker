"""Distinct OMI tipologias from features JSONL."""

from __future__ import annotations

import json
from pathlib import Path

from ml.train import DEFAULT_INPUT


def list_tipologie(path: Path = DEFAULT_INPUT) -> list[str]:
    """Distinct tipologias from features JSONL (empty if file missing)."""
    tips: set[str] = set()
    if path.is_file():
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    tip = json.loads(line).get("tipologia")
                    if isinstance(tip, str) and tip.strip():
                        tips.add(tip.strip())
        except (OSError, json.JSONDecodeError):
            pass
    return sorted(tips)
