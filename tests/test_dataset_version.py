"""Unit tests for RF-12 dataset fingerprinting."""

from __future__ import annotations

import json
from pathlib import Path

from ml.dataset_version import fingerprint
from ml.train import train


def _write_features(path: Path, n: int = 20) -> None:
    rows = []
    for i in range(n):
        day = "2026-09-01" if i < n // 2 else "2026-09-08"
        rows.append(
            {
                "listing_id": i,
                "scraped_at": f"{day}T10:00:00+00:00",
                "price_per_m2_monthly": 20.0 + i * 0.5,
                "surface_m2": 50.0 + i,
                "rooms": 2,
                "distance_from_center_km": 3.0,
                "area_price_per_m2_hist": 22.0,
                "publication_month": 9,
                "municipio": "I" if i % 2 == 0 else "II",
            }
        )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_fingerprint_stable_for_same_content(tmp_path: Path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    payload = '{"x": 1}\n{"x": 2}\n'
    a.write_text(payload, encoding="utf-8")
    b.write_text(payload, encoding="utf-8")
    fa = fingerprint(a)
    fb = fingerprint(b)
    assert fa["sha256"] == fb["sha256"]
    assert fa["n_rows"] == 2
    assert fa["nbytes"] == len(payload.encode("utf-8"))


def test_fingerprint_changes_when_content_changes(tmp_path: Path):
    path = tmp_path / "f.jsonl"
    path.write_text('{"x": 1}\n', encoding="utf-8")
    h1 = fingerprint(path)["sha256"]
    path.write_text('{"x": 2}\n', encoding="utf-8")
    h2 = fingerprint(path)["sha256"]
    assert h1 != h2


def test_train_writes_dataset_json(tmp_path: Path):
    features = tmp_path / "features.jsonl"
    _write_features(features, n=20)
    models_dir = tmp_path / "models"
    out = train(input_path=features, models_dir=models_dir, tracking_uri=None)

    for folder in (out, models_dir / "baseline_latest"):
        ds_path = folder / "dataset.json"
        assert ds_path.is_file()
        meta = json.loads(ds_path.read_text(encoding="utf-8"))
        assert "sha256" in meta and len(meta["sha256"]) == 64
        assert meta["n_rows"] == 20
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["dataset"]["sha256"] == meta["sha256"]
