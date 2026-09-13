"""Tests for admin OMI CSV ingest (token + SHA-256 dedup)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from api.ingest import IngestError, ingest_omi_csv, sync_manifest_from_disk
from api.main import create_app
from ml.train import FEATURE_COLS, PipelineBuilder
from tests.omi_rows import omi_feature_row
from tests.test_omi import _CSV_2024_1, _CSV_2024_2


def _tiny_model(tmp_path: Path) -> Path:
    rows = [omi_feature_row(i, day="2026-09-08", loc_mid_lag=15.0 + i) for i in range(24)]
    X = pd.DataFrame([{c: r.get(c) for c in FEATURE_COLS} for r in rows])
    y = [float(r["price_per_m2_monthly"]) for r in rows]
    pipe = PipelineBuilder().build()
    pipe.fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    return path


def test_ingest_dedup_by_content_hash(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    raw = tmp_path / "omi"
    content = _CSV_2024_1.encode("utf-8")
    first = ingest_omi_csv(
        filename="quotazioni_roma_2024_1.csv",
        content=content,
        raw_dir=raw,
        run_pipeline=False,
    )
    assert first.status == "stored"
    assert first.saved_as == "quotazioni_roma_2024_1.csv"
    assert (raw / first.saved_as).is_file()

    second = ingest_omi_csv(
        filename="quotazioni_roma_2024_1_copy.csv",
        content=content,
        raw_dir=raw,
        run_pipeline=False,
    )
    assert second.status == "duplicate"
    assert second.duplicate_of == "quotazioni_roma_2024_1.csv"
    assert not (raw / "quotazioni_roma_2024_1_copy.csv").exists()


def test_ingest_dedup_against_existing_disk_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    raw = tmp_path / "omi"
    raw.mkdir()
    path = raw / "quotazioni_roma_2024_2.csv"
    content = _CSV_2024_2.encode("utf-8")
    path.write_bytes(content)
    sync_manifest_from_disk(raw)

    result = ingest_omi_csv(
        filename="another_name.csv",
        content=content,
        raw_dir=raw,
        run_pipeline=False,
    )
    assert result.status == "duplicate"
    assert result.duplicate_of == "quotazioni_roma_2024_2.csv"


def test_ingest_rejects_bad_csv(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    try:
        ingest_omi_csv(
            filename="bad.csv",
            content=b"not,an,omi,file\n1,2,3\n",
            raw_dir=tmp_path / "omi",
        )
        raise AssertionError("expected IngestError")
    except IngestError as exc:
        assert "Invalid OMI CSV" in str(exc)


def test_ingest_endpoint_auth_and_duplicate(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("INGEST_TOKEN", "test-secret-token")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    raw_dir = tmp_path / "ingest_omi"
    monkeypatch.setattr("api.ingest.RAW_OMI_DIR", raw_dir)

    model = _tiny_model(tmp_path)
    client = TestClient(create_app(model))
    headers = {"X-Ingest-Token": "test-secret-token"}
    files = {
        "file": ("quotazioni_roma_2024_1.csv", _CSV_2024_1.encode("utf-8"), "text/csv")
    }
    data = {"run_pipeline": "false"}

    r1 = client.post("/ingest/omi", headers=headers, files=files, data=data)
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1["status"] == "stored"
    assert body1["n_rows"] and body1["n_rows"] > 0

    r2 = client.post("/ingest/omi", headers=headers, files=files, data=data)
    assert r2.status_code == 200
    assert r2.json()["status"] == "duplicate"

    bad = client.post(
        "/ingest/omi",
        headers={"X-Ingest-Token": "wrong"},
        files=files,
        data=data,
    )
    assert bad.status_code == 401

    monkeypatch.delenv("INGEST_TOKEN", raising=False)
    client2 = TestClient(create_app(model))
    disabled = client2.post(
        "/ingest/omi",
        headers={"X-Ingest-Token": "test-secret-token"},
        files=files,
        data=data,
    )
    assert disabled.status_code == 503


def test_ingest_cloud_upload_and_dispatch(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    calls: dict[str, object] = {"upload": 0, "dispatch": 0, "remote_files": {}}

    def fake_cloud_configured() -> bool:
        return True

    def fake_load_remote() -> dict:
        return {"version": 1, "files": dict(calls["remote_files"])}  # type: ignore[arg-type]

    def fake_save_remote(manifest: dict) -> None:
        calls["remote_files"] = dict(manifest.get("files") or {})

    def fake_upload(*, digest: str, filename: str, content: bytes) -> str:
        calls["upload"] = int(calls["upload"]) + 1
        return f"omi-ingest/files/{digest}/{filename}"

    def fake_dispatch_configured() -> bool:
        return True

    def fake_dispatch() -> str:
        calls["dispatch"] = int(calls["dispatch"]) + 1
        return "dispatched daily_monitoring.yml@main"

    monkeypatch.setattr("api.cloud_store.cloud_configured", fake_cloud_configured)
    monkeypatch.setattr("api.cloud_store.load_remote_manifest", fake_load_remote)
    monkeypatch.setattr("api.cloud_store.save_remote_manifest", fake_save_remote)
    monkeypatch.setattr("api.cloud_store.upload_csv", fake_upload)
    monkeypatch.setattr("api.github_dispatch.dispatch_configured", fake_dispatch_configured)
    monkeypatch.setattr("api.github_dispatch.dispatch_omi_monitoring", fake_dispatch)

    raw = tmp_path / "omi"
    result = ingest_omi_csv(
        filename="quotazioni_roma_2024_1.csv",
        content=_CSV_2024_1.encode("utf-8"),
        raw_dir=raw,
        run_pipeline=True,
    )
    assert result.status == "stored"
    assert result.cloud_key and result.cloud_key.startswith("omi-ingest/")
    assert result.workflow and "dispatched" in result.workflow
    assert calls["upload"] == 1
    assert calls["dispatch"] == 1

    dup = ingest_omi_csv(
        filename="copy.csv",
        content=_CSV_2024_1.encode("utf-8"),
        raw_dir=raw,
        run_pipeline=True,
    )
    assert dup.status == "duplicate"
    assert calls["upload"] == 1
    assert calls["dispatch"] == 1


def test_ensure_features_latest_from_dvc_remote(tmp_path: Path, monkeypatch):
    from api.cloud_store import dvc_cache_key, ensure_features_latest, read_dvc_md5

    dest = tmp_path / "features_latest.jsonl"
    pointer = tmp_path / "features_latest.jsonl.dvc"
    pointer.write_text(
        "outs:\n- md5: abcd1234ef567890abcd1234ef567890\n  path: features_latest.jsonl\n",
        encoding="utf-8",
    )
    assert read_dvc_md5(pointer) == "abcd1234ef567890abcd1234ef567890"
    assert dvc_cache_key("abcd1234ef567890abcd1234ef567890") == (
        "dvc/files/md5/ab/cd1234ef567890abcd1234ef567890"
    )

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "y")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://example.r2.cloudflarestorage.com")

    class _Body:
        def read(self) -> bytes:
            return b'{"zona_omi":"B12"}\n'

    class _Client:
        def get_object(self, Bucket: str, Key: str):
            assert Bucket == "rent-tracker-data"
            assert Key == "dvc/files/md5/ab/cd1234ef567890abcd1234ef567890"
            return {"Body": _Body()}

    monkeypatch.setattr("api.cloud_store._client", lambda: _Client())
    assert ensure_features_latest(dest, dvc_path=pointer) is True
    assert dest.read_text(encoding="utf-8").startswith('{"zona_omi"')
    assert ensure_features_latest(dest, dvc_path=pointer) is True


def test_ensure_features_latest_skips_without_cloud(tmp_path: Path, monkeypatch):
    from api.cloud_store import ensure_features_latest

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    dest = tmp_path / "features_latest.jsonl"
    pointer = tmp_path / "features_latest.jsonl.dvc"
    pointer.write_text("outs:\n- md5: abcd1234ef567890abcd1234ef567890\n", encoding="utf-8")
    assert ensure_features_latest(dest, dvc_path=pointer) is False
    assert not dest.exists()
