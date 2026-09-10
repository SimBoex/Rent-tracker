"""Decide whether to retrain when MAE exceeds a threshold (RF-09 lite)."""

from __future__ import annotations

from typing import Any
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import logging

from ml.drift_report import REPORTS_DIR, generate_drift_report
from ml.train import DEFAULT_INPUT, DEFAULT_TRACKING_URI, MODELS_DIR, train

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPORTS_DIR / "drift_latest" / "summary.json"

DEFAULT_MAE_RATIO = 1.5
DEFAULT_MIN_REFERENCE = 50
FEATURES_STABLE_SHARE = 0.5

logger = logging.getLogger(__name__)


class RetrainGate:
    """MAE-only retrain gate (aligned with doc/drift.md)."""

    def __init__(
        self,
        mae_ratio_threshold: float = DEFAULT_MAE_RATIO,
        min_reference: int = DEFAULT_MIN_REFERENCE,
    ):
        self.mae_ratio_threshold = mae_ratio_threshold
        self.min_reference = min_reference

    # summary is written by drift_report.py
    def decide(self, summary: dict[str, Any]) -> dict[str, Any]:
        mae_ref = summary.get("mae_reference")
        mae_cur = summary.get("mae_current")
        n_reference = int(summary.get("n_reference") or 0)
        drifted_share = summary.get("drifted_columns_share")
        features_stable = (
            drifted_share is not None and float(drifted_share) < FEATURES_STABLE_SHARE
        )

        base: dict[str, Any] = {
            "should_retrain": False,
            "trigger_reason": "missing_mae",
            "mae_ratio": None,
            "mae_ratio_threshold": self.mae_ratio_threshold,
            "min_reference": self.min_reference,
            "n_reference": n_reference,
            "mae_reference": mae_ref,
            "mae_current": mae_cur,
            "drifted_columns_share": drifted_share,
            "features_stable": features_stable,
        }

        if not isinstance(mae_ref, dict) or not isinstance(mae_cur, dict):
            return base
        ref_mae = mae_ref.get("mae")
        cur_mae = mae_cur.get("mae")
        if ref_mae is None or cur_mae is None:
            return base
        ref_mae_f = float(ref_mae)
        cur_mae_f = float(cur_mae)
        if ref_mae_f <= 0:
            base["trigger_reason"] = "missing_mae"
            return base

        ratio = cur_mae_f / ref_mae_f
        base["mae_ratio"] = round(ratio, 4)

        if n_reference < self.min_reference:
            base["trigger_reason"] = "insufficient_reference"
            return base

        if ratio >= self.mae_ratio_threshold:
            base["should_retrain"] = True
            base["trigger_reason"] = "mae_ratio"
            return base

        base["trigger_reason"] = "below_threshold"
        return base


def _write_decision(decision: dict[str, Any], reports_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = reports_dir / f"retrain_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(decision, ensure_ascii=False, indent=2) + "\n"
    (out_dir / "decision.json").write_text(payload, encoding="utf-8")

    latest = reports_dir / "retrain_latest"
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "decision.json").write_text(payload, encoding="utf-8")
    return out_dir


def run_retrain_check(
    summary_path: Path = DEFAULT_SUMMARY,
    reports_dir: Path = REPORTS_DIR,
    mae_ratio_threshold: float = DEFAULT_MAE_RATIO,
    min_reference: int = DEFAULT_MIN_REFERENCE,
    dry_run: bool = False,
    run_drift: bool = False,
    input_path: Path = DEFAULT_INPUT,
    models_dir: Path = MODELS_DIR,
    tracking_uri: str | None = DEFAULT_TRACKING_URI,
) -> dict[str, Any]:
    if run_drift:
        drift_dir = generate_drift_report(input_path=input_path, reports_dir=reports_dir)
        summary_path = reports_dir / "drift_latest" / "summary.json"
        logger.info("Drift refreshed → %s", drift_dir)

    if not summary_path.is_file():
        raise FileNotFoundError(f"Drift summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gate = RetrainGate(
        mae_ratio_threshold=mae_ratio_threshold,
        min_reference=min_reference,
    )
    decision = gate.decide(summary)
    decision["generated_at"] = datetime.now(timezone.utc).isoformat()
    decision["summary_path"] = str(summary_path)
    decision["dry_run"] = dry_run
    decision["model_path"] = None

    if decision["should_retrain"] and not dry_run:
        out = train(
            input_path=input_path,
            models_dir=models_dir,
            tracking_uri=tracking_uri,
        )
        decision["model_path"] = str(out)
        logger.info("Retrain triggered → %s", out)
    elif decision["should_retrain"] and dry_run:
        logger.info("Retrain would run (dry-run); reason=%s", decision["trigger_reason"])
    else:
        logger.info("No retrain; reason=%s", decision["trigger_reason"])

    out_dir = _write_decision(decision, reports_dir)
    logger.info("Retrain decision → %s", out_dir)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RF-09: retrain if mae_current/mae_reference exceeds threshold."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument(
        "--mae-ratio",
        type=float,
        default=DEFAULT_MAE_RATIO,
        help=f"Retrain if mae_current/mae_reference >= this (default {DEFAULT_MAE_RATIO})",
    )
    parser.add_argument(
        "--min-reference",
        type=int,
        default=DEFAULT_MIN_REFERENCE,
        help=f"Skip trigger if n_reference below this (default {DEFAULT_MIN_REFERENCE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Decide and write decision.json without calling train",
    )
    parser.add_argument(
        "--run-drift",
        action="store_true",
        help="Regenerate drift report before deciding",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_retrain_check(
        summary_path=args.summary,
        reports_dir=args.reports_dir,
        mae_ratio_threshold=args.mae_ratio,
        min_reference=args.min_reference,
        dry_run=args.dry_run,
        run_drift=args.run_drift,
        input_path=args.input,
        models_dir=args.models_dir,
        tracking_uri=None if args.no_mlflow else DEFAULT_TRACKING_URI,
    )


if __name__ == "__main__":
    main()
