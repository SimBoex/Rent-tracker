"""Lightweight public snapshot loaders for Render UI (httpx/json only — no joblib)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_GOOD_DEALS = REPORTS_DIR / "good_deals_latest.json"
DEFAULT_MONITORING = REPORTS_DIR / "monitoring_latest.json"
DEFAULT_GOOD_DEALS_URL = (
    "https://raw.githubusercontent.com/SimBoex/Rent-tracker/main/reports/good_deals_latest.json"
)
DEFAULT_MONITORING_URL = (
    "https://raw.githubusercontent.com/SimBoex/Rent-tracker/main/reports/monitoring_latest.json"
)

logger = logging.getLogger(__name__)


def _fetch_json(fetch_url: str) -> dict[str, Any] | None:
    import httpx

    try:
        resp = httpx.get(fetch_url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Could not fetch %s: %s", fetch_url, exc)
        return None


def load_good_deals_snapshot(
    path: Path = DEFAULT_GOOD_DEALS,
    url: str | None = None,
) -> dict[str, Any] | None:
    """Load precomputed good deals from disk, else from URL (Render UI)."""
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    fetch_url = (
        url if url is not None else os.environ.get("GOOD_DEALS_URL", DEFAULT_GOOD_DEALS_URL)
    ).strip()
    if not fetch_url:
        return None
    return _fetch_json(fetch_url)


def load_monitoring_snapshot(
    path: Path = DEFAULT_MONITORING,
    url: str | None = None,
) -> dict[str, Any] | None:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    fetch_url = (
        url if url is not None else os.environ.get("MONITORING_URL", DEFAULT_MONITORING_URL)
    ).strip()
    if not fetch_url:
        return None
    return _fetch_json(fetch_url)
