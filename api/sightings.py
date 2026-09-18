from pathlib import Path
import json
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_DIR = ROOT / "data" / "raw" / "sightings"
DEFAULT_PATH = DEFAULT_DIR / "sightings.jsonl"



def append_sighting(path: Path, record: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path




