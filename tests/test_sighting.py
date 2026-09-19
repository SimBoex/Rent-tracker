
from __future__ import annotations
import json
from api.sightings import append_sighting, sighting_digest, check_seen_sighting
from pathlib import Path
import hashlib

from fastapi.testclient import TestClient

from api.main import create_app
from tests.test_api import _tiny_model

def test_append_sighting_writes_two_rows(tmp_path: Path) -> None:
    path = tmp_path / "sightings.jsonl"
    append_sighting(path, {"sighting_id": "id-1", "zona_omi": "B12", "tipologia": "Abitazioni civili", "stato": "NORMALE", "asking_eur_m2": 1000})
    append_sighting(path, {"sighting_id": "id-2", "zona_omi": "B12", "tipologia": "Abitazioni civili", "stato": "NORMALE", "asking_eur_m2": 1000})
    assert len(path.read_text().splitlines()) == 1 # only one row


def test_sighting_digest() -> None:
    assert sighting_digest("B12", "Abitazioni civili", "NORMALE", 1000) == sighting_digest("B12", "Abitazioni civili", "NORMALE", 1000)
    assert sighting_digest("B12", "Abitazioni civili", "NORMALE", 1000) !=  sighting_digest("B12", "Abitazioni civili", "NORMALE", 1001)
    assert sighting_digest(" B12 ", "Abitazioni civili", "NORMALE", 1000) ==  sighting_digest("B12", "Abitazioni civili", "NORMALE", 1000)


def test_sighting_upload(tmp_path: Path, monkeypatch):
    def fake_cloud_configured() -> bool:
        return False

    monkeypatch.setattr("api.sightings.cloud_configured", fake_cloud_configured)

    sightings = tmp_path / "sightings.jsonl"
    features = tmp_path / "features.jsonl"
    rows = [
        {
            "zona_omi": "B12",
            "tipologia": "Abitazioni civili",
            "stato": "NORMALE",
            "semester": "2025-1",
            "price_per_m2_monthly": 10,
        }
    ]
    features.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    model_path = _tiny_model(tmp_path)
    payload = {
        "zona_omi": "B12",
        "tipologia": "Abitazioni civili",
        "stato": "NORMALE",
        "asking_eur_m2": 10,
    }
    with TestClient(create_app(model_path, features_path=features, sightings_path=sightings)) as client:
        resp1 = client.post("/sightings", json=payload)
        assert resp1.status_code == 200
        body1 = resp1.json()
        assert body1["sighting_id"] is not None
        assert body1["status"] == "ok"
        assert body1["duplicate_of"] is None
        assert sightings.is_file()
        assert len(sightings.read_text().splitlines()) == 1

        first_id = body1["sighting_id"]
        resp2 = client.post("/sightings", json=payload)
        assert resp2.status_code == 200
        body2 = resp2.json()
        assert body2["status"] == "duplicate"
        assert body2["duplicate_of"] == first_id
        assert len(sightings.read_text().splitlines()) == 1
