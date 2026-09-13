"""Admin OMI CSV ingest: token auth, SHA-256 dedup (local + R2), optional CI dispatch."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from etl.extract.omi_loader import RAW_OMI_DIR, load_omi_csv

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = ".ingest_manifest.json"
MAX_BYTES = 50 * 1024 * 1024
_SAFE_NAME = re.compile(r"^[\w.\-]+\.csv$", re.IGNORECASE)

logger = logging.getLogger(__name__)


class IngestError(Exception):
    """Validation / policy failure (maps to 4xx)."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class IngestResult:
    status: str
    sha256: str
    saved_as: str | None
    n_rows: int | None
    semester: str | None
    duplicate_of: str | None = None
    pipeline: str | None = None
    cloud_key: str | None = None
    workflow: str | None = None


def ingest_token() -> str | None:
    tok = os.environ.get("INGEST_TOKEN", "").strip()
    return tok or None


def _manifest_path(raw_dir: Path) -> Path:
    return raw_dir / MANIFEST_NAME


def _load_manifest(raw_dir: Path) -> dict[str, Any]:
    path = _manifest_path(raw_dir)
    if not path.is_file():
        return {"version": 1, "files": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "files": {}}
    if not isinstance(data, dict):
        return {"version": 1, "files": {}}
    files = data.get("files")
    if not isinstance(files, dict):
        data["files"] = {}
    return data


def _save_manifest(raw_dir: Path, manifest: dict[str, Any]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(raw_dir)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sync_manifest_from_disk(raw_dir: Path) -> dict[str, Any]:
    """Index existing CSVs by content hash so pre-UI files also dedupe."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(raw_dir)
    files: dict[str, Any] = dict(manifest.get("files") or {})
    known_names = {str(v.get("filename")) for v in files.values() if isinstance(v, dict)}

    for path in sorted({*raw_dir.glob("*.csv"), *raw_dir.glob("*.CSV")}):
        if path.name.startswith("."):
            continue
        if path.name in known_names:
            continue
        try:
            digest = sha256_bytes(path.read_bytes())
        except OSError:
            continue
        if digest in files:
            continue
        files[digest] = {
            "filename": path.name,
            "nbytes": path.stat().st_size,
            "uploaded_at": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "source": "disk_sync",
        }
    manifest["files"] = files
    _save_manifest(raw_dir, manifest)
    return manifest


def sanitize_filename(name: str) -> str:
    base = Path(name).name.strip()
    if not base or not _SAFE_NAME.match(base):
        raise IngestError(
            "Invalid filename: use a simple *.csv name "
            "(letters, digits, _.- only), e.g. QI_…_20252_VALORI.csv"
        )
    return base


def _dest_path(raw_dir: Path, filename: str, digest: str) -> Path:
    candidate = raw_dir / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    return raw_dir / f"{stem}_{digest[:8]}{suffix}"


def _run_pipeline_skip_train() -> str:
    cmd = [sys.executable, str(ROOT / "run_pipeline.py"), "--skip-train", "-v"]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return "run_pipeline --skip-train ok"


def _merge_remote_into_local(local: dict[str, Any], remote: dict[str, Any]) -> dict[str, Any]:
    files = dict(local.get("files") or {})
    for digest, meta in (remote.get("files") or {}).items():
        if digest not in files and isinstance(meta, dict):
            files[digest] = meta
    local["files"] = files
    return local


def ingest_omi_csv(
    *,
    filename: str,
    content: bytes,
    raw_dir: Path | None = None,
    run_pipeline: bool = False,
) -> IngestResult:
    from api.cloud_store import (
        cloud_configured,
        load_remote_manifest,
        save_remote_manifest,
        upload_csv,
    )
    from api.github_dispatch import dispatch_configured, dispatch_omi_monitoring

    if not content:
        raise IngestError("Empty file")
    if len(content) > MAX_BYTES:
        raise IngestError(f"File too large (max {MAX_BYTES // (1024 * 1024)} MB)")

    safe_name = sanitize_filename(filename)
    if not safe_name.lower().endswith(".csv"):
        raise IngestError("Only .csv uploads are accepted")

    digest = sha256_bytes(content)
    dest_dir = raw_dir or RAW_OMI_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest = sync_manifest_from_disk(dest_dir)

    use_cloud = cloud_configured()
    remote: dict[str, Any] | None = None
    if use_cloud:
        try:
            remote = load_remote_manifest()
            manifest = _merge_remote_into_local(manifest, remote)
        except Exception as exc:
            raise IngestError(
                f"Cloud store unreachable: {exc}",
                status_code=503,
            ) from exc

    existing = (manifest.get("files") or {}).get(digest)
    if isinstance(existing, dict):
        return IngestResult(
            status="duplicate",
            sha256=digest,
            saved_as=None,
            n_rows=None,
            semester=None,
            duplicate_of=str(existing.get("filename") or digest),
        )

    # Validate by parsing a temp file whose name still carries semester hints.
    tmp = dest_dir / f".upload_{digest[:12]}_{safe_name}"
    try:
        tmp.write_bytes(content)
        rows = load_omi_csv(tmp)
    except ValueError as exc:
        tmp.unlink(missing_ok=True)
        raise IngestError(f"Invalid OMI CSV: {exc}") from exc
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    if not rows:
        tmp.unlink(missing_ok=True)
        raise IngestError("CSV parsed but produced 0 residential locazione rows")

    semester = str(rows[0].get("semester") or "") or None
    final = _dest_path(dest_dir, safe_name, digest)
    tmp.replace(final)

    entry = {
        "filename": final.name,
        "nbytes": len(content),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows),
        "semester": semester,
        "source": "ingest",
    }
    manifest.setdefault("files", {})[digest] = entry
    _save_manifest(dest_dir, manifest)

    cloud_key = None
    if use_cloud:
        try:
            cloud_key = upload_csv(digest=digest, filename=final.name, content=content)
            entry["key"] = cloud_key
            entry["source"] = "ingest_cloud"
            manifest["files"][digest] = entry
            _save_manifest(dest_dir, manifest)
            remote_out = remote or {"version": 1, "files": {}}
            remote_out.setdefault("files", {})[digest] = entry
            save_remote_manifest(remote_out)
        except Exception as exc:
            raise IngestError(
                f"Saved locally as {final.name} but cloud upload failed: {exc}",
                status_code=502,
            ) from exc

    pipeline_msg = None
    workflow_msg = None
    if run_pipeline:
        if use_cloud and dispatch_configured():
            # Cloud: durable bytes are on R2 — let GitHub Actions rebuild features/drift.
            try:
                workflow_msg = dispatch_omi_monitoring()
            except Exception as exc:
                raise IngestError(
                    f"Uploaded to cloud but workflow_dispatch failed: {exc}",
                    status_code=502,
                ) from exc
        elif use_cloud and not dispatch_configured():
            pipeline_msg = (
                "stored on R2; set GITHUB_TOKEN + GITHUB_REPOSITORY to auto-run "
                "omi-monitoring (or trigger the workflow manually)"
            )
        else:
            try:
                pipeline_msg = _run_pipeline_skip_train()
            except subprocess.CalledProcessError as exc:
                raise IngestError(
                    f"Saved {final.name} but pipeline failed (exit {exc.returncode})",
                    status_code=500,
                ) from exc

    return IngestResult(
        status="stored",
        sha256=digest,
        saved_as=final.name,
        n_rows=len(rows),
        semester=semester,
        pipeline=pipeline_msg,
        cloud_key=cloud_key,
        workflow=workflow_msg,
    )
