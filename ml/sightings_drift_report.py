from etl.jsonl import load_jsonl
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any
from statistics import mean
import argparse
import json
from api.cloud_store import download_sightings_jsonl, cloud_configured

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
SIGHTINGS_PATH = ROOT / "data/raw/sightings/sightings.jsonl"

class SightingsDriftReport:
    def __init__(self, path: Path):
        self.path = path
        self.data = load_jsonl(path)


    def build(self, reports_dir: Path) -> dict:

        reference_rows, current_rows = self.temporal_split(self.data)

        metrics = self.extract_metrics(current_rows)
        self.save(metrics, reports_dir)
        return metrics

    def temporal_split(self, data: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        
        current_rows = []
        reference_rows = []
        for entry in data:
            submitted_at = datetime.fromisoformat(entry["submitted_at"])
            if submitted_at.tzinfo is None:
                submitted_at = submitted_at.replace(tzinfo=timezone.utc)
            if submitted_at >= datetime.now(timezone.utc) - timedelta(days=30):
                current_rows.append(entry)
            else:
                reference_rows.append(entry)
        
        if len(current_rows) < 20:
            raise ValueError("Not enough samples in the current period")

        return reference_rows, current_rows

    def extract_metrics(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        mae = []
        bias = []

        for row in rows:
            predicted_price_per_m2_monthly = row["predicted_price_per_m2_monthly"]
            asking_eur_m2 = row["asking_eur_m2"]
            drift =  asking_eur_m2 - predicted_price_per_m2_monthly
            mae.append(abs(drift))
            bias.append(drift)

        mae = mean(mae)
        bias = mean(bias)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input": str(self.path),
            "n_samples": len(rows),
            "mae": mae,
            "bias": bias
        }

    def save(self, metrics: dict[str, Any], reports_dir: Path) -> None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = reports_dir / f"SightingsDriftReport_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_path = out_dir / "summary.json"

        summary_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        latest_dir = reports_dir / "sightings_drift_latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        (latest_dir / "summary.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sightings drift report (asking vs fair).")
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Download sightings.jsonl from R2 before building the report",
    )
    parser.add_argument("--input", type=Path, default=SIGHTINGS_PATH)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    path = args.input
    need_pull = args.pull or not path.is_file() or path.stat().st_size == 0
    if need_pull:
        if not cloud_configured():
            raise SystemExit(
                f"Sightings file missing/empty at {path} and cloud is not configured "
                "(set AWS_* or pass a local --input)."
            )
        if not download_sightings_jsonl(path):
            raise SystemExit(f"Failed to pull sightings.jsonl from R2 → {path}")

    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"No sightings data at {path}")

    SightingsDriftReport(path).build(args.reports_dir)


if __name__ == "__main__":
    main()