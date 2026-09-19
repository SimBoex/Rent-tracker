from pathlib import Path
import json
from typing import Any
import logging
from api.cloud_store import cloud_configured, append_remote_sighting_line
import hashlib
from datetime import datetime, timezone
from api.cloud_store import load_remote_seen_sightings, save_remote_seen_sightings

logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_DIR = ROOT / "data" / "raw" / "sightings"
DEFAULT_PATH = DEFAULT_DIR / "sightings.jsonl"


# this is called from the API endpoint
def append_sighting(path: Path, record: dict[str, Any]) -> tuple[str, str|None]:
    digest = sighting_digest(record["zona_omi"], record["tipologia"], record["stato"], record["asking_eur_m2"])

    if cloud_configured():
        seen_dict = load_remote_seen_sightings()
        if digest in seen_dict.get("digests", {}):
            logger.info("Sighting already seen: %s", digest)
            return ("duplicate", seen_dict["digests"][digest]["sighting_id"])

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            # save locally
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        append_remote_sighting_line(record)
        seen_dict["digests"][digest]={"sighting_id": record["sighting_id"], "submitted_at": datetime.now(timezone.utc).isoformat()}
        save_remote_seen_sightings(seen_dict)
        path_seen = path.parent / "seen.json"
        check_seen_sighting(digest, record["sighting_id"], path_seen)
        return ("ok", None)
            
    else:
        seen_path = path.parent / "seen.json"
        seen_path.parent.mkdir(parents=True, exist_ok=True)
        if check_seen_sighting(digest, record["sighting_id"], seen_path):
            logger.info("Sighting already seen: %s", digest)
            seen_dict = load_seen_sightings(seen_path)
            return ("duplicate", seen_dict["digests"][digest]["sighting_id"])

        with path.open("a", encoding="utf-8") as fh:
            # save locally
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        
        return ("ok", None)


def sighting_digest(zona_omi: str, tipologia:str, stato:str, asking_eur_m2: float) -> str:
    return hashlib.sha256(f"{zona_omi.strip()}|{tipologia.strip()}|{stato.strip()}|{float(asking_eur_m2):.2f}".encode("utf-8")).hexdigest()


def check_seen_sighting(digest : str, sighting_id: str, seen_path: Path) -> bool:
    seen_dict = load_seen_sightings(seen_path)
    if digest in seen_dict.get("digests", {}):
        return True
    else:  
        
        seen_dict["digests"][digest]={"sighting_id": sighting_id, "submitted_at": datetime.now(timezone.utc).isoformat()}
        # append to seen.json
        seen_path.write_text(json.dumps(seen_dict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return False



def load_seen_sightings(seen_path: Path) -> dict:
    if not seen_path.is_file():
        return {"version": 1, "digests": {}}
    return json.loads(seen_path.read_text(encoding="utf-8"))

