"""Unit tests for RF-09 retrain gate."""

from __future__ import annotations

import json
from pathlib import Path

from ml.retrain_check import RetrainGate, run_retrain_check


def _summary(
    *,
    mae_ref: float | None = 4.0,
    mae_cur: float | None = 7.0,
    n_reference: int = 80,
    drifted_share: float = 0.25,
) -> dict:
    out: dict = {
        "n_reference": n_reference,
        "drifted_columns_share": drifted_share,
        "mae_reference": None if mae_ref is None else {"n": n_reference, "mae": mae_ref, "rmse": mae_ref},
        "mae_current": None if mae_cur is None else {"n": 100, "mae": mae_cur, "rmse": mae_cur},
    }
    return out


def test_decide_triggers_on_mae_ratio():
    d = RetrainGate().decide(_summary(mae_ref=4.0, mae_cur=7.0))
    assert d["should_retrain"] is True
    assert d["trigger_reason"] == "mae_ratio"
    assert d["mae_ratio"] == 1.75
    assert d["features_stable"] is True


def test_decide_below_threshold():
    d = RetrainGate().decide(_summary(mae_ref=4.0, mae_cur=5.0))
    assert d["should_retrain"] is False
    assert d["trigger_reason"] == "below_threshold"
    assert d["mae_ratio"] == 1.25


def test_decide_insufficient_reference():
    d = RetrainGate().decide(_summary(mae_ref=4.0, mae_cur=8.0, n_reference=20))
    assert d["should_retrain"] is False
    assert d["trigger_reason"] == "insufficient_reference"
    assert d["mae_ratio"] == 2.0


def test_decide_missing_mae():
    d = RetrainGate().decide(_summary(mae_ref=None, mae_cur=None))
    assert d["should_retrain"] is False
    assert d["trigger_reason"] == "missing_mae"
    assert d["mae_ratio"] is None


def test_decide_features_not_stable_still_triggers_on_mae():
    d = RetrainGate().decide(
        _summary(mae_ref=4.0, mae_cur=7.0, drifted_share=0.75)
    )
    assert d["should_retrain"] is True
    assert d["features_stable"] is False


def test_dry_run_writes_decision(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(_summary(mae_ref=4.0, mae_cur=7.0)) + "\n",
        encoding="utf-8",
    )
    reports_dir = tmp_path / "reports"
    decision = run_retrain_check(
        summary_path=summary_path,
        reports_dir=reports_dir,
        dry_run=True,
    )
    assert decision["should_retrain"] is True
    assert decision["dry_run"] is True
    assert decision["model_path"] is None
    latest = reports_dir / "retrain_latest" / "decision.json"
    assert latest.is_file()
    saved = json.loads(latest.read_text(encoding="utf-8"))
    assert saved["trigger_reason"] == "mae_ratio"
