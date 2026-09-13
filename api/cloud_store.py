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


def object_key(digest: str, filename: str) -> str:
    return f"{INGEST_PREFIX}/files/{digest}/{filename}"


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
    key = object_key(digest, filename)
    client = _client()
    client.put_object(
        Bucket=_bucket(),
        Key=key,
        Body=content,
        ContentType="text/csv",
    )
    logger.info("Uploaded ingest object s3://%s/%s", _bucket(), key)
    return key


def download_inbox(raw_dir) -> list[str]:
    """Download all CSV objects listed in the remote manifest into raw_dir.

    Skips hashes already present as local files with the same content name.
    Returns list of filenames written.
    """
    from pathlib import Path

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
        key = str(meta.get("key") or object_key(digest, filename))
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
