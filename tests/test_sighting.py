
from __future__ import annotations
import json
from api.sightings import append_sighting
from pathlib import Path


from fastapi.testclient import TestClient

from api.main import create_app
from tests.test_api import _tiny_model

def test_append_sighting_writes_two_rows(tmp_path: Path) -> None:
    path = tmp_path / "sightings.jsonl"
    append_sighting(path, {"zone": "B12", "tipologia": "Abitazioni civili", "stato": "NORMALE", "asking_eur_m2": 1000})
    append_sighting(path, {"zone": "B12", "tipologia": "Abitazioni civili", "stato": "NORMALE", "asking_eur_m2": 1000})
    assert path.read_text(encoding="utf-8") == json.dumps({"zone": "B12", "tipologia": "Abitazioni civili", "stato": "NORMALE", "asking_eur_m2": 1000}) + "\n" + json.dumps({"zone": "B12", "tipologia": "Abitazioni civili", "stato": "NORMALE", "asking_eur_m2": 1000}) + "\n"




def test_sighting_upload(tmp_path: Path, monkeypatch):
    calls: dict[str, object] = {"upload": 0}
    def fake_cloud_configured() -> bool:
        return True
    
    def fake_upload_sighting(*, sighting_id: str, content: bytes) -> None:
        calls["upload"] += 1
        return f"sightings/{sighting_id}.json"

    monkeypatch.setattr("api.main.cloud_configured", fake_cloud_configured)
    monkeypatch.setattr("api.main.upload_sighting", fake_upload_sighting)
    
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
    with TestClient(create_app(model_path, features_path=features, sightings_path=sightings)) as client:
        resp = client.post("/sightings", json={
            "zona_omi": "B12",
            "tipologia": "Abitazioni civili",
            "stato": "NORMALE",
            "asking_eur_m2": 10,
        })
        assert resp.status_code == 200
        assert resp.json()["sighting_id"] is not None
        assert calls["upload"] == 1

        assert sightings.is_file()
        assert len(sightings.read_text().splitlines()) == 1

    
