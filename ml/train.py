"""Train a baseline regressor for €/m²/month (RF-05) with MLflow tracking (RF-11 lite)."""

from __future__ import annotations

from typing import Any
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import logging
import math

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ml.dataset_version import fingerprint, write_dataset_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "features_latest.jsonl"
MODELS_DIR = ROOT / "models"
MLFLOW_DB = ROOT / "mlflow.db"
DEFAULT_TRACKING_URI = f"sqlite:///{MLFLOW_DB.resolve()}"

TARGET = "price_per_m2_monthly"
# OMI Open Data features — see doc/omi.md / SOR2 RF-01d
NUMERIC_FEATURES = ["loc_mid_lag", "publication_month"]
CATEGORICAL_FEATURES = ["zona_omi", "tipologia", "stato"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

HOLDOUT_FRAC = 0.2

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(
        self,
        input_path: Path = DEFAULT_INPUT,
        require_key: str = "zona_omi",
    ):
        self.input_path = input_path
        self.require_key = require_key

    def load(self) -> list[dict[str, Any]]:
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input not found: {self.input_path}")

        rows: list[dict[str, Any]] = []
        dropped_target = 0
        dropped_key = 0
        for line in self.input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get(TARGET) is None:
                dropped_target += 1
                continue
            key_val = row.get(self.require_key)
            if key_val is None or key_val == "":
                dropped_key += 1
                continue
            rows.append(row)
        logger.info(
            "Loaded %s rows from %s (dropped missing target=%s, missing %s=%s)",
            len(rows),
            self.input_path,
            dropped_target,
            self.require_key,
            dropped_key,
        )
        return rows


class PipelineBuilder:
    def __init__(
        self,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ):
        self.numeric_features = (
            list(NUMERIC_FEATURES) if numeric_features is None else list(numeric_features)
        )
        self.categorical_features = (
            list(CATEGORICAL_FEATURES)
            if categorical_features is None
            else list(categorical_features)
        )
        self.feature_cols = self.numeric_features + self.categorical_features

    def build(self) -> Pipeline:
        transformers: list[tuple] = []
        if self.numeric_features:
            transformers.append(("num", "passthrough", self.numeric_features))
        transformers.append(
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                self.categorical_features,
            )
        )
        pre = ColumnTransformer(transformers=transformers)
        # After transform: [numeric..., categorical...]
        n_num = len(self.numeric_features)
        cat_idx = list(range(n_num, n_num + len(self.categorical_features)))
        return Pipeline(
            steps=[
                ("pre", pre),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        random_state=42,
                        categorical_features=cat_idx,  # pyright: ignore[reportArgumentType]
                    ),
                ),
            ]
        )



class MetricsCalculator:
    def evaluate(self, y_true: list[float], y_pred: list[float]) -> dict[str, float]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
        return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def _scraped_day(row: dict[str, Any]) -> str:
    raw = str(row.get("scraped_at") or "")
    return raw[:10] if len(raw) >= 10 else ""


def _listing_sort_key(row: dict[str, Any]) -> tuple[str, str, int | str]:
    lid = row.get("listing_id")
    if isinstance(lid, int):
        lid_key: int | str = lid
    elif isinstance(lid, str) and lid.isdigit():
        lid_key = int(lid)
    else:
        lid_key = str(lid or "")
    return (_scraped_day(row), str(row.get("scraped_at") or ""), lid_key)


class Trainer:
    def __init__(
        self,
        pipeline: Pipeline,
        metrics_calculator: MetricsCalculator,
        data_loader: DataLoader,
        feature_cols: list[str] | None = None,
    ):
        self.pipeline: Pipeline = pipeline
        self.metrics_calculator: MetricsCalculator = metrics_calculator
        self.data_loader: DataLoader = data_loader
        self.feature_cols = feature_cols or list(FEATURE_COLS)
        self.metrics: dict[str, Any] = {}
        self.split_mode: str = ""
        self.n_rows: int = 0
        self.n_train: int = 0
        self.n_test: int = 0

    def split_data(
        self,
    ) -> tuple[pd.DataFrame, list[float], pd.DataFrame, list[float]]:
        rows = self.data_loader.load()
        train_rows, test_rows, split_mode = self.temporal_or_ordered_split(rows)
        if not train_rows or not test_rows:
            raise ValueError(
                f"Split produced empty set (train={len(train_rows)}, test={len(test_rows)})"
            )

        self.split_mode = split_mode
        self.n_rows = len(rows)
        self.n_train = len(train_rows)
        self.n_test = len(test_rows)
        if split_mode == "ordered_holdout_fallback":
            logger.warning(
                "Only one scrape day — using ordered_holdout_fallback (last %.0f%%); not a true temporal split",
                100.0 * HOLDOUT_FRAC,
            )
        X_train, y_train = self.extract_features_and_target(train_rows)
        X_test, y_test = self.extract_features_and_target(test_rows)
        return X_train, y_train, X_test, y_test

    def temporal_or_ordered_split(
        self,
        rows: list[dict[str, Any]],
        holdout_frac: float = HOLDOUT_FRAC,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        """Train/test split: last scrape day as test if ≥2 days; else ordered holdout."""
        if not rows:
            return [], [], "empty"

        ordered = sorted(rows, key=_listing_sort_key)
        days = sorted({_scraped_day(r) for r in ordered if _scraped_day(r)})

        if len(days) >= 2:
            last = days[-1]
            train = [r for r in ordered if _scraped_day(r) != last]
            test = [r for r in ordered if _scraped_day(r) == last]
            if train and test:
                return train, test, "temporal_last_day"

        n = len(ordered)
        n_test = max(1, int(round(n * holdout_frac)))
        if n_test >= n:
            n_test = max(1, n // 5) if n >= 5 else 1
        split_at = n - n_test
        train, test = ordered[:split_at], ordered[split_at:]
        return train, test, "ordered_holdout_fallback"

    def extract_features_and_target(
        self, rows: list[dict[str, Any]]
    ) -> tuple[pd.DataFrame, list[float]]:
        X = pd.DataFrame([{c: r.get(c) for c in self.feature_cols} for r in rows])
        y = [float(r[TARGET]) for r in rows]
        return X, y

    def fit(self, X_train: pd.DataFrame, y_train: list[float]) -> Pipeline:
        self.pipeline.fit(X_train, y_train)
        return self.pipeline

    def predict(self, X_test: pd.DataFrame) -> list[float]:
        y_pred = self.pipeline.predict(X_test)
        return list(y_pred)

    def evaluate(self, y_test: list[float], y_pred: list[float]) -> dict[str, Any]:
        scores = self.metrics_calculator.evaluate(y_test, y_pred)
        self.metrics = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input": str(self.data_loader.input_path),
            "model": "HistGradientBoostingRegressor",
            "features": self.feature_cols,
            "target": TARGET,
            "split_mode": self.split_mode,
            "n_rows": self.n_rows,
            "n_train": self.n_train,
            "n_test": self.n_test,
            **scores,
        }
        if self.split_mode == "ordered_holdout_fallback":
            self.metrics["split_warning"] = (
                "Single scraped_at day; holdout is ordered by (scraped_at, listing_id), not temporal."
            )
        return self.metrics

    def save_model(self, model_path: Path) -> None:
        joblib.dump(self.pipeline, model_path)

    def save_metrics(self, metrics_path: Path) -> None:
        metrics_path.write_text(
            json.dumps(self.metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def run(
        self,
        models_dir: Path = MODELS_DIR,
        tracking_uri: str | None = DEFAULT_TRACKING_URI,
    ) -> Path:
        """Fit baseline, write model.joblib + metrics.json, log to local MLflow."""
        X_train, y_train, X_test, y_test = self.split_data()
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        metrics = self.evaluate(y_test, y_pred)
        dataset_meta = fingerprint(self.data_loader.input_path)
        metrics["dataset"] = dataset_meta
        self.metrics = metrics
        scores = {k: float(metrics[k]) for k in ("mae", "rmse", "r2")}

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = models_dir / f"baseline_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.joblib"
        metrics_path = out_dir / "metrics.json"
        dataset_path = out_dir / "dataset.json"

        self.save_model(model_path)
        self.save_metrics(metrics_path)
        write_dataset_json(dataset_meta, dataset_path)

        latest_dir = models_dir / "baseline_latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        self.save_model(latest_dir / "model.joblib")
        self.save_metrics(latest_dir / "metrics.json")
        write_dataset_json(dataset_meta, latest_dir / "dataset.json")

        logger.info(
            "Test MAE=%.3f RMSE=%.3f R2=%.3f (split=%s, n_train=%s, n_test=%s, dataset_sha256=%s) → %s",
            scores["mae"],
            scores["rmse"],
            scores["r2"],
            self.split_mode,
            self.n_train,
            self.n_test,
            dataset_meta["sha256"][:12],
            out_dir,
        )

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("roma-rent-baseline")
            with mlflow.start_run(run_name=f"baseline_{run_id}"):
                mlflow.log_params(
                    {
                        "model": "HistGradientBoostingRegressor",
                        "split_mode": self.split_mode,
                        "n_train": self.n_train,
                        "n_test": self.n_test,
                        "features": ",".join(self.feature_cols),
                        "dataset_sha256": dataset_meta["sha256"],
                    }
                )
                mlflow.log_metrics(scores)
                mlflow.log_artifact(str(model_path))
                mlflow.log_artifact(str(metrics_path))
                mlflow.log_artifact(str(dataset_path))

        return out_dir


def _numeric_cols_with_signal(rows: list[dict[str, Any]], cols: list[str]) -> list[str]:
    """Drop numeric cols with <2 distinct finite values (HGB binning would crash)."""
    keep: list[str] = []
    for col in cols:
        uniq: set[float] = set()
        for row in rows:
            val = row.get(col)
            if val is None:
                continue
            try:
                f = float(val)
            except (TypeError, ValueError):
                continue
            if math.isnan(f):
                continue
            uniq.add(f)
            if len(uniq) >= 2:
                keep.append(col)
                break
    return keep


def train(
    input_path: Path = DEFAULT_INPUT,
    models_dir: Path = MODELS_DIR,
    tracking_uri: str | None = DEFAULT_TRACKING_URI,
) -> Path:
    loader = DataLoader(input_path)
    rows = loader.load()
    # HGB needs ≥2 distinct finite values *on the train fold* (not the full dataset).
    # With 2 semesters, train is often only the earlier one → lag all-null / month constant.
    probe = Trainer(
        pipeline=PipelineBuilder().build(),
        metrics_calculator=MetricsCalculator(),
        data_loader=loader,
    )
    train_rows, _test_rows, _mode = probe.temporal_or_ordered_split(rows)
    numeric = _numeric_cols_with_signal(train_rows, list(NUMERIC_FEATURES))
    dropped = [c for c in NUMERIC_FEATURES if c not in numeric]
    if dropped:
        logger.warning(
            "Dropping numeric features without signal on train fold: %s "
            "(need ≥3 OMI semesters for lag/month on temporal train)",
            dropped,
        )
    builder = PipelineBuilder(numeric, list(CATEGORICAL_FEATURES))
    trainer = Trainer(
        pipeline=builder.build(),
        metrics_calculator=MetricsCalculator(),
        data_loader=loader,
        feature_cols=list(builder.feature_cols),
    )
    return trainer.run(models_dir=models_dir, tracking_uri=tracking_uri)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline rent €/m² model (RF-05).")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--no-mlflow", action="store_true", help="Skip local MLflow logging")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    train(
        input_path=args.input,
        models_dir=args.models_dir,
        tracking_uri=None if args.no_mlflow else DEFAULT_TRACKING_URI,
    )


if __name__ == "__main__":
    main()

