"""Generate data + prediction drift reports with Evidently (RF-08 lite)."""

from __future__ import annotations

from typing import Any
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import logging

import joblib
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from ml.train import (
    DEFAULT_INPUT,
    FEATURE_COLS,
    MODELS_DIR,
    TARGET,
    DataLoader,
    MetricsCalculator,
    PipelineBuilder,
    Trainer,
    _scraped_day,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
DEFAULT_MODEL = MODELS_DIR / "baseline_latest" / "model.joblib"
PREDICTION_COL = "prediction"

logger = logging.getLogger(__name__)


def _scores_from_frame(frame: pd.DataFrame) -> dict[str, Any] | None:
    """MAE/RMSE on a split frame that already has prediction."""
    if PREDICTION_COL not in frame.columns:
        return None
    scores = MetricsCalculator().evaluate(
        frame[TARGET].tolist(),
        frame[PREDICTION_COL].tolist(),
    )
    return {"n": int(len(frame)), "mae": scores["mae"], "rmse": scores["rmse"]}


class DriftFrameBuilder:
    """Build reference/current frames for Evidently (features + optional prediction)."""

    def __init__(
        self,
        data_loader: DataLoader,
        model_path: Path | None = DEFAULT_MODEL,
    ):
        self.data_loader = data_loader
        self.model_path = model_path
        self.split_mode: str = ""
        self.n_reference: int = 0
        self.n_current: int = 0
        self.mae_by_day: list[dict[str, Any]] | None = None

    # Build reference/current frames for Evidently (features + optional prediction).
    def build(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows = self.data_loader.load()

        splitter = Trainer(
            pipeline=PipelineBuilder().build(),
            metrics_calculator=MetricsCalculator(),
            data_loader=self.data_loader,
        )
        reference_rows, current_rows, split_mode = splitter.temporal_or_ordered_split(rows)
        if not reference_rows or not current_rows:
            raise ValueError(
                f"Drift split empty (reference={len(reference_rows)}, current={len(current_rows)})"
            )

        self.split_mode = split_mode
        self.n_reference = len(reference_rows)
        self.n_current = len(current_rows)
        if split_mode == "ordered_holdout_fallback":
            logger.warning(
                "Only one scrape day — using ordered_holdout_fallback for reference/current"
            )

        ref = self._rows_to_frame(reference_rows)
        cur = self._rows_to_frame(current_rows)

        # Add prediction column
        if self.model_path is not None and self.model_path.is_file():
            model = joblib.load(self.model_path)
            ref[PREDICTION_COL] = model.predict(ref[FEATURE_COLS])
            cur[PREDICTION_COL] = model.predict(cur[FEATURE_COLS])
            self.mae_by_day = self._mae_by_day(rows, model)
            logger.info("Added %s from %s", PREDICTION_COL, self.model_path)
        else:
            self.mae_by_day = None
            logger.warning(
                "Model not found at %s — data/target drift only (no prediction column)",
                self.model_path,
            )

        return ref, cur

    def _rows_to_frame(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        records = [{c: r.get(c) for c in FEATURE_COLS} for r in rows]
        frame = pd.DataFrame(records)
        frame[TARGET] = [float(r[TARGET]) for r in rows]
        return frame

    def _mae_by_day(
        self, rows: list[dict[str, Any]], model: Any
    ) -> list[dict[str, Any]]:
        """Per-scrape-day MAE/RMSE — proxy for P(y|x) / concept drift."""
        by_day: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            day = _scraped_day(row) or "unknown"
            by_day.setdefault(day, []).append(row)

        calc = MetricsCalculator()
        out: list[dict[str, Any]] = []
        for day in sorted(by_day):
            day_rows = by_day[day]
            frame = self._rows_to_frame(day_rows)
            y_true = frame[TARGET].tolist()
            y_pred = list(model.predict(frame[FEATURE_COLS]))
            scores = calc.evaluate(y_true, y_pred)
            out.append(
                {
                    "day": day,
                    "n": len(day_rows),
                    "mae": scores["mae"],
                    "rmse": scores["rmse"],
                }
            )
        return out


class DriftReporter:
    def __init__(self, frame_builder: DriftFrameBuilder):
        self.frame_builder = frame_builder

    def run(self, reports_dir: Path = REPORTS_DIR) -> Path:
        reference, current = self.frame_builder.build()
        snapshot = Report([DataDriftPreset()]).run(current, reference)

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = reports_dir / f"drift_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        html_path = out_dir / "report.html"
        summary_path = out_dir / "summary.json"
        snapshot.save_html(str(html_path))

        summary = self._build_summary(snapshot, reference, current)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        latest_dir = reports_dir / "drift_latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(latest_dir / "report.html"))
        (latest_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        logger.info(
            "Drift report → %s (split=%s, n_ref=%s, n_cur=%s, drifted_share=%s)",
            out_dir,
            summary.get("split_mode"),
            summary.get("n_reference"),
            summary.get("n_current"),
            summary.get("drifted_columns_share"),
        )
        return out_dir

    def _build_summary(
        self,
        snapshot: Any,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> dict[str, Any]:
        metrics = snapshot.dict().get("metrics", [])
        drifted = next(
            (m for m in metrics if "DriftedColumnsCount" in str(m.get("metric_name", ""))),
            None,
        )
        value = (drifted or {}).get("value") or {}
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input": str(self.frame_builder.data_loader.input_path),
            "model_path": str(self.frame_builder.model_path)
            if self.frame_builder.model_path
            else None,
            "split_mode": self.frame_builder.split_mode,
            "n_reference": self.frame_builder.n_reference,
            "n_current": self.frame_builder.n_current,
            "columns": list(current.columns),
            "has_prediction": PREDICTION_COL in current.columns,
            "drifted_columns_count": value.get("count"),
            "drifted_columns_share": value.get("share"),
            "mae_reference": _scores_from_frame(reference),
            "mae_current": _scores_from_frame(current),
            "mae_by_day": self.frame_builder.mae_by_day,
            "reference_rows": int(len(reference)),
            "current_rows": int(len(current)),
        }


def generate_drift_report(
    input_path: Path = DEFAULT_INPUT,
    model_path: Path | None = DEFAULT_MODEL,
    reports_dir: Path = REPORTS_DIR,
) -> Path:
    reporter = DriftReporter(
        frame_builder=DriftFrameBuilder(
            data_loader=DataLoader(input_path),
            model_path=model_path,
        )
    )
    return reporter.run(reports_dir=reports_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evidently data/prediction drift report (RF-08)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip loading baseline model (no prediction column)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    generate_drift_report(
        input_path=args.input,
        model_path=None if args.no_model else args.model,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
