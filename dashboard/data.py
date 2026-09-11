"""Load monitoring JSON and score OMI zone rows for the Streamlit dashboard (RF-10)."""

from __future__ import annotations

from typing import Any
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import logging

import pandas as pd

from api.predictor import BELOW_OMI_BAND, DEFAULT_MODEL_PATH, ModelPredictor
from dashboard.snapshots import (
    DEFAULT_GOOD_DEALS,
    DEFAULT_MONITORING,
    load_good_deals_snapshot,
    load_monitoring_snapshot,
)
from ml.drift_report import REPORTS_DIR
from ml.train import DEFAULT_INPUT, FEATURE_COLS, TARGET

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIFT_SUMMARY = REPORTS_DIR / "drift_latest" / "summary.json"
DEFAULT_RETRAIN_DECISION = REPORTS_DIR / "retrain_latest" / "decision.json"
MAX_SCORE_ROWS = 2000
logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DRIFT_SUMMARY",
    "DEFAULT_RETRAIN_DECISION",
    "DEFAULT_GOOD_DEALS",
    "DEFAULT_MONITORING",
    "export_good_deals",
    "export_monitoring_snapshot",
    "good_deals_table",
    "load_feature_rows",
    "load_good_deals_snapshot",
    "load_json",
    "load_monitoring_snapshot",
    "score_rows",
]

DISPLAY_COLS = [
    "zona_omi",
    "tipologia",
    "stato",
    "semester",
    TARGET,
    "predicted_price_per_m2_monthly",
    "gap_pct",
    "deal_label",
]

# Public / Render snapshot: no OMI €/m² (mid) and no gap (would reveal mid).
# Labels + our model prediction only; always cite Agenzia Entrate – OMI.
SOURCE_ATTRIBUTION = "Agenzia Entrate – OMI"
PUBLIC_DISPLAY_COLS = [
    "zona_omi",
    "tipologia",
    "stato",
    "semester",
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
        zona = row.get("zona_omi")
        if zona is None or zona == "":
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def score_rows(
    rows: list[dict[str, Any]],
    predictor: ModelPredictor,
) -> pd.DataFrame:
    scored: list[dict[str, Any]] = []
    for row in rows:
        features = {c: row.get(c) for c in FEATURE_COLS}
        result = predictor.score(features, actual_price_per_m2=float(row[TARGET]))
        scored.append(
            {
                "zona_omi": row.get("zona_omi"),
                "tipologia": row.get("tipologia"),
                "stato": row.get("stato"),
                "semester": row.get("semester"),
                TARGET: float(row[TARGET]),
                "predicted_price_per_m2_monthly": result["predicted_price_per_m2_monthly"],
                "gap_pct": result["gap_pct"],
                "deal_label": result["deal_label"],
            }
        )
    if not scored:
        return pd.DataFrame(columns=DISPLAY_COLS)
    return pd.DataFrame(scored)


# Back-compat name for callers
score_listings = score_rows


def good_deals_table(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty or "deal_label" not in scored.columns:
        return pd.DataFrame(columns=DISPLAY_COLS)
    deals = scored[scored["deal_label"] == BELOW_OMI_BAND].copy()
    return deals.sort_values("gap_pct", ascending=True).reset_index(drop=True)


def export_good_deals(
    features_path: Path = DEFAULT_INPUT,
    model_path: Path = DEFAULT_MODEL_PATH,
    out_path: Path = DEFAULT_GOOD_DEALS,
    limit: int = MAX_SCORE_ROWS,
) -> Path:
    """Score OMI features and write a public JSON snapshot for the UI."""
    rows = load_feature_rows(features_path, limit=limit)
    predictor = ModelPredictor(model_path)
    scored = score_rows(rows, predictor)
    deals = good_deals_table(scored)
    public = deals.reindex(columns=PUBLIC_DISPLAY_COLS)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_attribution": SOURCE_ATTRIBUTION,
        "note": (
            "Public snapshot: zone labels + deal_label only. "
            "No OMI locazione €/m² values (mid/min/max) or residuals."
        ),
        "n_scored": int(len(scored)),
        "n_good_deals": int(len(public)),
        "columns": list(PUBLIC_DISPLAY_COLS),
        "rows": public.to_dict(orient="records"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s below-band rows → %s (scored=%s)", len(public), out_path, len(scored))
    return out_path


def _public_drift(drift: dict[str, Any] | None) -> dict[str, Any] | None:
    if not drift:
        return None
    skip = {"input", "model_path", "columns"}
    return {k: v for k, v in drift.items() if k not in skip}


def export_monitoring_snapshot(
    drift_path: Path = DEFAULT_DRIFT_SUMMARY,
    decision_path: Path = DEFAULT_RETRAIN_DECISION,
    out_path: Path = DEFAULT_MONITORING,
) -> Path:
    drift = load_json(drift_path)
    decision = load_json(decision_path)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_attribution": SOURCE_ATTRIBUTION,
        "drift": _public_drift(drift),
        "decision": decision,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote monitoring snapshot → %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export public dashboard snapshots (zone deals + monitoring)."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out-deals", type=Path, default=DEFAULT_GOOD_DEALS)
    parser.add_argument("--out-monitoring", type=Path, default=DEFAULT_MONITORING)
    parser.add_argument("--limit", type=int, default=MAX_SCORE_ROWS)
    parser.add_argument(
        "--monitoring-only",
        action="store_true",
        help="Skip deals export (requires drift/retrain JSON already present)",
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
