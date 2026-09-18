"""S3/R2 helpers for durable OMI ingest (Render → object storage → CI)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

INGEST_PREFIX = "omi-ingest"
MANIFEST_KEY = f"{INGEST_PREFIX}/manifest.json"
SIGHTINGS_PREFIX = "sightings-inbox"

SIGHTINGS_MANIFEST_KEY = f"{SIGHTINGS_PREFIX}/manifest.json"

def cloud_configured() -> bool:
    """True when S3/R2 credentials are present (bucket defaults to rent-tracker-data)."""
    return bool(
        os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
        and os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
    )


def _bucket() -> str:
    return (
        os.environ.get("INGEST_S3_BUCKET", "").strip()
        or os.environ.get("AWS_S3_BUCKET", "").strip()
        or "rent-tracker-data"
    )


def _client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 required for cloud ingest — pip install boto3"
        ) from exc

    kwargs: dict[str, Any] = {
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"].strip(),
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"].strip(),
    }
    endpoint = os.environ.get("AWS_ENDPOINT_URL", "").strip()
    region = os.environ.get("AWS_DEFAULT_REGION", "").strip() or "auto"
    if endpoint:
        kwargs["endpoint_url"] = endpoint
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)

# path for a file in the bucket
def object_key(ingest_prefix: str, digest: str, filename: str) -> str:
    return f"{ingest_prefix}/files/{digest}/{filename}"


def load_remote_manifest() -> dict[str, Any]:
    client = _client()
    bucket = _bucket()
    try:
        obj = client.get_object(Bucket=bucket, Key=MANIFEST_KEY)
        body = obj["Body"].read().decode("utf-8")
        data = json.loads(body)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return {"version": 1, "files": {}}
        raise
    if not isinstance(data, dict):
        return {"version": 1, "files": {}}
    if not isinstance(data.get("files"), dict):
        data["files"] = {}
    return data

def load_remote_sightings_manifest() -> dict[str, Any]:
    client = _client()
    bucket = _bucket()
    try:
        obj = client.get_object(Bucket=bucket, Key=SIGHTINGS_MANIFEST_KEY)
        body = obj["Body"].read().decode("utf-8")
        data = json.loads(body)
    except ClientError as exc:
        raise
    if not isinstance(data, dict):
        return {"version": 1, "files": {}}
    if not isinstance(data.get("files"), dict):
        data["files"] = {}
    return data

def save_remote_sightings_manifest(manifest: dict[str, Any]) -> None:
    client = _client()
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    client.put_object(
        Bucket=_bucket(),
        Key=SIGHTINGS_MANIFEST_KEY,
        Body=payload.encode("utf-8"),
        ContentType="application/json",
    )

def save_remote_manifest(manifest: dict[str, Any]) -> None:
    client = _client()
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    client.put_object(
        Bucket=_bucket(),
        Key=MANIFEST_KEY,
        Body=payload.encode("utf-8"),
        ContentType="application/json",
    )


def upload_csv(*, digest: str, filename: str, content: bytes) -> str:
    key = object_key("omi-ingest", digest, filename)
    client = _client()
    client.put_object(
        Bucket=_bucket(),
        Key=key,
        Body=content,
        ContentType="text/csv",
    )
    logger.info("Uploaded ingest object s3://%s/%s", _bucket(), key)
    return key

def upload_sighting(*, sighting_id: str, content: bytes) -> str:
    key = f"sightings-inbox/{sighting_id}.json"
    client = _client()
    client.put_object(
        Bucket=_bucket(),
        Key=key,
        Body=content,
        ContentType="application/json",
    )
    logger.info("Uploaded sighting object s3://%s/%s", _bucket(), key)
    return key

from pathlib import Path


def download_sightings(raw_dir: Path) -> list[str]:
    """Download all sightings objects listed in the remote manifest into raw_dir.
       return a list of sighting ids files; 
    """

    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    if not cloud_configured():
        logger.warning("Cloud ingest not configured — skip sightings pull")
        return []
    
    sightings_manifest = load_remote_sightings_manifest()
    sightings = sightings_manifest.get("files") or {}
    client = _client()
    bucket = _bucket()
    written: list[str] = []

    for sighting_id, meta in sightings.items():
        filename = str(meta.get("filename") or "")
        key = str(meta.get("key") or object_key("sightings-inbox", sighting_id, filename))
        if not filename or not filename.lower().endswith(".json"):
            continue
        dest = raw / filename
        if dest.is_file():
            # Same name already on disk (e.g. from dvc pull) — keep existing
            continue
        obj = client.get_object(Bucket=bucket, Key=key)
        dest.write_bytes(obj["Body"].read())
        written.append(filename)
        logger.info("Sighting → %s", dest)
    return written


def download_inbox(raw_dir) -> list[str]:
    """Download all CSV objects listed in the remote manifest into raw_dir.

    Skips hashes already present as local files with the same content name.
    Returns list of filenames written.
    """

    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    if not cloud_configured():
        logger.warning("Cloud ingest not configured — skip inbox pull")
        return []

    manifest = load_remote_manifest()
    files = manifest.get("files") or {}
    client = _client()
    bucket = _bucket()
    written: list[str] = []

    for digest, meta in files.items():
        if not isinstance(meta, dict):
            continue
        filename = str(meta.get("filename") or "")
        key = str(meta.get("key") or object_key("omi-ingest", digest, filename))
        if not filename or not filename.lower().endswith(".csv"):
            continue
        dest = raw / filename
        if dest.is_file():
            # Same name already on disk (e.g. from dvc pull) — keep existing
            continue
        obj = client.get_object(Bucket=bucket, Key=key)
        dest.write_bytes(obj["Body"].read())
        written.append(filename)
        logger.info("Inbox → %s", dest)
    return written


def read_dvc_md5(dvc_path) -> str | None:
    """Parse md5 from a DVC pointer file (``*.dvc``)."""
    from pathlib import Path

    path = Path(dvc_path)
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().lstrip("-").strip()
        if stripped.startswith("md5:"):
            digest = stripped.split(":", 1)[1].strip()
            return digest or None
    return None


def dvc_cache_key(md5: str, remote_prefix: str = "dvc") -> str:
    """Object key for a DVC md5 cache entry under the remote prefix."""
    digest = md5.strip().lower()
    prefix = remote_prefix.strip().strip("/") or "dvc"
    return f"{prefix}/files/md5/{digest[:2]}/{digest[2:]}"


def ensure_features_latest(
    dest,
    dvc_path=None,
    *,
    remote_prefix: str = "dvc",
) -> bool:
    """Ensure ``features_latest.jsonl`` exists locally; pull from R2/DVC if needed.

    Returns True when the file is present after the call.
    Uses the same AWS_* credentials as ingest (Render already has them for DVC).
    """
    from pathlib import Path

    out = Path(dest)
    if out.is_file() and out.stat().st_size > 0:
        return True

    pointer = Path(dvc_path) if dvc_path is not None else Path(str(out) + ".dvc")
    md5 = read_dvc_md5(pointer)
    if not md5:
        logger.warning("No DVC md5 at %s — cannot fetch features", pointer)
        return False
    if not cloud_configured():
        logger.warning("Cloud not configured — cannot fetch features from R2")
        return False

    key = dvc_cache_key(md5, remote_prefix=remote_prefix)
    client = _client()
    bucket = _bucket()
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
        payload = obj["Body"].read()
    except ClientError as exc:
        logger.warning("Failed to download s3://%s/%s: %s", bucket, key, exc)
        return False
    if not payload:
        logger.warning("Empty features object at s3://%s/%s", bucket, key)
        return False

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    logger.info("Features ← s3://%s/%s → %s (%s bytes)", bucket, key, out, len(payload))
    return True
