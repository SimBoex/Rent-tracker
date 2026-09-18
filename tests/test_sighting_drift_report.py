from ml.sightings_drift_report import SightingsDriftReport
from pathlib import Path
import json
from datetime import datetime, timezone




def test_sightings_drift_report(tmp_path: Path):
    sightings = tmp_path / "sightings.jsonl"
    _write_sightings(sightings)
    report = SightingsDriftReport(sightings)
    metrics = report.build(reports_dir = tmp_path / "reports")
    assert metrics["mae"] == 0
    assert metrics["bias"] == 0
    assert metrics["n_samples"] == 21

    assert (tmp_path / "reports" / "sightings_drift_latest" / "summary.json").is_file()



def _write_sightings(path: Path, n: int = 21) -> None:
    rows = []
    for i in range(n):
        rows.append({"submitted_at": datetime.now(timezone.utc).isoformat(), "predicted_price_per_m2_monthly": 1000, "asking_eur_m2": 1000})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

