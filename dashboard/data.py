"""Load monitoring JSON and score listings for the Streamlit dashboard (RF-10)."""

from __future__ import annotations

from typing import Any
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import logging
import os

import pandas as pd

from api.predictor import DEFAULT_MODEL_PATH, GOOD_DEAL, ModelPredictor
from ml.drift_report import REPORTS_DIR
from ml.train import DEFAULT_INPUT, FEATURE_COLS, TARGET

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIFT_SUMMARY = REPORTS_DIR / "drift_latest" / "summary.json"
DEFAULT_RETRAIN_DECISION = REPORTS_DIR / "retrain_latest" / "decision.json"
DEFAULT_GOOD_DEALS = REPORTS_DIR / "good_deals_latest.json"
DEFAULT_MONITORING = REPORTS_DIR / "monitoring_latest.json"
# Public UI (Render) can fetch committed snapshots without local features/model.
DEFAULT_GOOD_DEALS_URL = (
    "https://raw.githubusercontent.com/SimBoex/Rent-tracker/main/reports/good_deals_latest.json"
)
DEFAULT_MONITORING_URL = (
    "https://raw.githubusercontent.com/SimBoex/Rent-tracker/main/reports/monitoring_latest.json"
)
MAX_SCORE_ROWS = 2000
logger = logging.getLogger(__name__)

DISPLAY_COLS = [
    "listing_id",
    "url",
    "municipio",
    "surface_m2",
    "rooms",
    TARGET,
    "predicted_price_per_m2_monthly",
    "gap_pct",
    "deal_label",
]

# Public snapshot (RNF-03): no listing URLs / ids — demo metrics only.
PUBLIC_DISPLAY_COLS = [
    "municipio",
    "surface_m2",
    "rooms",
    TARGET,
    "predicted_price_per_m2_monthly",
    "gap_pct",
    "deal_label",
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_rows(path: Path = DEFAULT_INPUT, limit: int = MAX_SCORE_ROWS) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get(TARGET) is None:
            continue
        municipio = row.get("municipio")
        if municipio is None or municipio == "":
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def score_listings(
    rows: list[dict[str, Any]],
    predictor: ModelPredictor,
) -> pd.DataFrame:
    scored: list[dict[str, Any]] = []
    for row in rows:
        features = {c: row.get(c) for c in FEATURE_COLS}
        # predictor.score takes features and actual_price_per_m2 and returns a dictionary with the predicted price per m2 monthly, gap percentage, and deal label
        result = predictor.score(features, actual_price_per_m2=float(row[TARGET]))
        scored.append(
            {
                "listing_id": row.get("listing_id"),
                "url": row.get("url"),
                "municipio": row.get("municipio"),
                "surface_m2": row.get("surface_m2"),
                "rooms": row.get("rooms"),
                TARGET: float(row[TARGET]),
                "predicted_price_per_m2_monthly": result["predicted_price_per_m2_monthly"],
                "gap_pct": result["gap_pct"],
                "deal_label": result["deal_label"],
            }
        )
    if not scored:
        return pd.DataFrame(columns=DISPLAY_COLS)
    return pd.DataFrame(scored)


def good_deals_table(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty or "deal_label" not in scored.columns:
        return pd.DataFrame(columns=DISPLAY_COLS)
    deals = scored[scored["deal_label"] == GOOD_DEAL].copy()
    return deals.sort_values("gap_pct", ascending=True).reset_index(drop=True)


def export_good_deals(
    features_path: Path = DEFAULT_INPUT,
    model_path: Path = DEFAULT_MODEL_PATH,
    out_path: Path = DEFAULT_GOOD_DEALS,
    limit: int = MAX_SCORE_ROWS,
) -> Path:
    """Score features and write a privacy-safe JSON snapshot for the public UI (no urls/ids)."""
    rows = load_feature_rows(features_path, limit=limit)
    predictor = ModelPredictor(model_path)
    scored = score_listings(rows, predictor)
    deals = good_deals_table(scored)
    public = deals.reindex(columns=PUBLIC_DISPLAY_COLS)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_scored": int(len(scored)),
        "n_good_deals": int(len(public)),
        "columns": list(PUBLIC_DISPLAY_COLS),
        "rows": public.to_dict(orient="records"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s good deals → %s (scored=%s)", len(public), out_path, len(scored))
    return out_path


def load_good_deals_snapshot(
    path: Path = DEFAULT_GOOD_DEALS,
    url: str | None = None,
) -> dict[str, Any] | None:
    """Load precomputed good deals from disk, else from URL (Render UI)."""
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    fetch_url = (url if url is not None else os.environ.get("GOOD_DEALS_URL", DEFAULT_GOOD_DEALS_URL)).strip()
    if not fetch_url:
        return None
    import httpx

    try:
        resp = httpx.get(fetch_url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Could not fetch good deals from %s: %s", fetch_url, exc)
        return None


def _public_drift(drift: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop machine paths; keep aggregate metrics only (safe to publish)."""
    if not drift:
        return None
    skip = {"input"}
    return {k: v for k, v in drift.items() if k not in skip}


def export_monitoring_snapshot(
    drift_path: Path = DEFAULT_DRIFT_SUMMARY,
    decision_path: Path = DEFAULT_RETRAIN_DECISION,
    out_path: Path = DEFAULT_MONITORING,
) -> Path:
    """Write aggregate drift + retrain metrics for the public UI."""
    drift = load_json(drift_path)
    decision = load_json(decision_path)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "drift": _public_drift(drift),
        "decision": decision,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote monitoring snapshot → %s", out_path)
    return out_path


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
    import httpx

    try:
        resp = httpx.get(fetch_url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Could not fetch monitoring from %s: %s", fetch_url, exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export public dashboard snapshots (good deals + monitoring)."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out-deals", type=Path, default=DEFAULT_GOOD_DEALS)
    parser.add_argument("--out-monitoring", type=Path, default=DEFAULT_MONITORING)
    parser.add_argument("--limit", type=int, default=MAX_SCORE_ROWS)
    parser.add_argument(
        "--monitoring-only",
        action="store_true",
        help="Skip good-deals export (requires drift/retrain JSON already present)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not args.monitoring_only:
        export_good_deals(
            features_path=args.features,
            model_path=args.model,
            out_path=args.out_deals,
            limit=args.limit,
        )
    export_monitoring_snapshot(out_path=args.out_monitoring)


if __name__ == "__main__":
    main()
