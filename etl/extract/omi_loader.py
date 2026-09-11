"""Extract: load Agenzia Entrate OMI quotazioni CSV into data/raw/omi normalized JSONL."""

from __future__ import annotations

import argparse
import csv
import io
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RAW_OMI_DIR = ROOT / "data" / "raw" / "omi"
DEFAULT_OUT = RAW_OMI_DIR / "omi_quotazioni_latest.jsonl"

logger = logging.getLogger(__name__)

# Logical field → accepted header aliases (upper)
ALIASES: dict[str, tuple[str, ...]] = {
    "comune": ("COMUNE_DESCRIZIONE", "COMUNE", "COMUNE_DESC"),
    "zona": ("ZONA", "COD_ZONA", "ZONA_OMI"),
    "zona_descr": ("ZONA_DESCR", "DESCRIZIONE_ZONA", "ZONA_DESCRIZIONE"),
    "tipologia": ("DESCR_TIPOLOGIA", "TIPOLOGIA", "DESCRIZIONE_TIPOLOGIA"),
    "stato": ("STATO", "STATO_CONSERVATIVO", "STATO_PREVALENTE"),
    "loc_min": ("LOCMIN", "LOC_MIN", "LOCAZIONE_MIN", "LOC_MINIMO"),
    "loc_max": ("LOCMAX", "LOC_MAX", "LOCAZIONE_MAX", "LOC_MASSIMO"),
    "semester": ("SEMESTRE", "SEM", "PERIODO"),
}

FILENAME_SEMESTER = re.compile(r"(20\d{2})[_-]?([12])", re.I)
# Official export titles: "Semestre 2025/2"
TITLE_SEMESTER = re.compile(r"SEMESTRE\s+(20\d{2})\s*[/_-]\s*([12])", re.I)
HEADER_MARKERS = ("LOC_MIN", "LOCMIN", "LOC_MAX", "LOCMAX")


def _norm_header(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().upper())


def _map_headers(fieldnames: list[str]) -> dict[str, str]:
    """logical → actual csv header."""
    by_norm = {_norm_header(h): h for h in fieldnames}
    mapping: dict[str, str] = {}
    for logical, aliases in ALIASES.items():
        for alias in aliases:
            if alias in by_norm:
                mapping[logical] = by_norm[alias]
                break
    return mapping


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip().replace(",", ".")
    if not s or s.upper() in {"NA", "N/A", "NULL", "-"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def semester_from_filename(path: Path) -> str | None:
    m = FILENAME_SEMESTER.search(path.stem)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def semester_from_title(line: str) -> str | None:
    m = TITLE_SEMESTER.search(line)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def _is_header_line(line: str) -> bool:
    norm = _norm_header(line.replace(";", " ").replace(",", " "))
    return any(marker in norm for marker in HEADER_MARKERS)


def _is_residential(tipologia: str | None) -> bool:
    if not tipologia:
        return True
    t = tipologia.lower()
    # Keep dwellings; drop obvious non-residential if present in mixed exports
    deny = ("negozio", "ufficio", "capannone", "laboratorio", "magazzino", "box ", "posto auto")
    return not any(d in t for d in deny)


def load_omi_csv(path: Path, default_semester: str | None = None) -> list[dict[str, Any]]:
    """Parse one OMI quotazioni CSV into normalized row dicts."""
    semester = default_semester or semester_from_filename(path)
    with path.open(encoding="utf-8-sig", newline="") as fh:
        lines = fh.readlines()

    # Official exports often start with a title line before the real header.
    header_idx = None
    for i, line in enumerate(lines[:20]):
        if semester is None:
            semester = semester_from_title(line)
        if _is_header_line(line):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"{path.name}: no OMI valori header (Loc_min/Loc_max) found")

    table = "".join(lines[header_idx:])
    try:
        dialect = csv.Sniffer().sniff(table[:4096], delimiters=";,\t")
    except csv.Error:
        dialect = csv.excel
        dialect.delimiter = ";"

    reader = csv.DictReader(io.StringIO(table), dialect=dialect)
    if not reader.fieldnames:
        raise ValueError(f"No header in {path}")
    mapping = _map_headers(list(reader.fieldnames))
    required = ("zona", "tipologia", "loc_min", "loc_max")
    missing = [k for k in required if k not in mapping]
    if missing:
        raise ValueError(f"{path.name}: missing columns for {missing}; got {reader.fieldnames}")

    rows: list[dict[str, Any]] = []
    for raw in reader:
        tip = (raw.get(mapping["tipologia"]) or "").strip() or None
        if not _is_residential(tip):
            continue
        loc_min = _parse_float(raw.get(mapping["loc_min"]))
        loc_max = _parse_float(raw.get(mapping["loc_max"]))
        if loc_min is None or loc_max is None or loc_min <= 0 or loc_max < loc_min:
            continue
        sem = semester
        if "semester" in mapping:
            cell = (raw.get(mapping["semester"]) or "").strip()
            if cell:
                sem = cell
        if not sem:
            raise ValueError(
                f"{path.name}: cannot infer semester — name file like "
                f"QI_*_20252_VALORI.csv / quotazioni_roma_2024_1.csv or add SEMESTRE column"
            )
        comune = None
        if "comune" in mapping:
            comune = (raw.get(mapping["comune"]) or "").strip() or None
        # Roma-only filter when comune present
        if comune and "roma" not in comune.lower():
            continue
        zona = (raw.get(mapping["zona"]) or "").strip()
        if not zona:
            continue
        zona_descr = None
        if "zona_descr" in mapping:
            zona_descr = (raw.get(mapping["zona_descr"]) or "").strip().strip("'") or None
        stato = None
        if "stato" in mapping:
            stato = (raw.get(mapping["stato"]) or "").strip() or None
        rows.append(
            {
                "source": "agenziaentrate-omi",
                "comune": comune or "Roma",
                "zona_omi": zona,
                "zona_omi_descr": zona_descr,
                "tipologia": tip,
                "stato": stato,
                "loc_min": loc_min,
                "loc_max": loc_max,
                "semester": sem,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "input_file": path.name,
            }
        )
    logger.info("Loaded %s rows from %s (semester=%s)", len(rows), path.name, semester)
    return rows


def load_omi_dir(raw_dir: Path = RAW_OMI_DIR) -> list[dict[str, Any]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = sorted({*raw_dir.glob("*.csv"), *raw_dir.glob("*.CSV")})
    # Prefer official VALORI exports; skip ZONE sidecars when both present.
    valori = [p for p in files if "VALORI" in p.name.upper()]
    candidates = valori if valori else files
    if not candidates:
        raise FileNotFoundError(
            f"No CSV in {raw_dir}. Download OMI quotazioni into data/raw/omi/ (see doc/omi.md)."
        )
    all_rows: list[dict[str, Any]] = []
    used = 0
    for path in candidates:
        try:
            rows = load_omi_csv(path)
        except ValueError as exc:
            logger.warning("Skipping %s (%s)", path.name, exc)
            continue
        all_rows.extend(rows)
        used += 1
    if not all_rows:
        raise ValueError(f"No OMI locazione rows loaded from {raw_dir}")
    logger.info("OMI total rows: %s from %s files", len(all_rows), used)
    return all_rows


def write_jsonl(rows: list[dict[str, Any]], out_path: Path = DEFAULT_OUT) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            import json

            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %s → %s", len(rows), out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Load OMI quotazioni CSV → JSONL.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_OMI_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    rows = load_omi_dir(args.raw_dir)
    write_jsonl(rows, args.out)


if __name__ == "__main__":
    main()
