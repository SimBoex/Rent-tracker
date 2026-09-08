"""transform phase: clean raw Immobiliare listings, dedupe, compute €/m² target."""

from __future__ import annotations

from typing import Any
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import logging
import json
import math
import re

from .constants import (
    MIN_SURFACE_M2,
    MAX_SURFACE_M2,
    MIN_PRICE_EUR,
    MAX_PRICE_EUR,
    ROME_CENTER_LAT,
    ROME_CENTER_LON,
    EARTH_RADIUS_KM,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


DROP_RATE_WARN = 0.25


logger = logging.getLogger(__name__)

    
class ListingTransformer:
    """Orchestration: load raw → dedupe → clean → write processed (like ImmobiliareScraper)."""

    def __init__(self, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR, drop_rate_warn: float = DROP_RATE_WARN) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.drop_rate_warn = drop_rate_warn
        self.last_quality: dict[str, Any] | None = None

    def run(self) -> Path:
        """Clean + quarantine, then persist artifacts."""
        raw_rows = self.load_raw_rows()
        deduped = self.dedupe_latest(raw_rows)
        cleaned, rejected = self.transform(deduped)
        return self.write_outputs(cleaned, rejected)
    
    def transform(self, deduped: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Clean + quarantine."""
        reasons: Counter[str] = Counter()
        cleaned: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for row in deduped:
            output, reason = self.clean_row(row)

            if output is None:
                reason_key = reason or "unknown"
                reasons[reason_key] += 1
                rejected.append({**row, "drop_reason": reason_key})
            else:
                cleaned.append(output)
        
        n_in = len(deduped)
        n_out = len(cleaned)
        n_drop = len(rejected)
        drop_rate = (n_drop / n_in) if n_in else 0.0

        self.last_quality = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_after_dedupe": n_in,
            "n_cleaned": n_out,
            "n_rejected": n_drop,
            "drop_rate": round(drop_rate, 4),
            "drop_rate_warn_threshold": self.drop_rate_warn,
            "drop_rate_alert": drop_rate >= self.drop_rate_warn,
            "by_reason": dict(reasons),
        }

        logger.info(
            "Cleaned rows: %s (dropped %s, drop_rate=%.1f%%) reasons=%s",
            n_out,
            n_drop,
            100.0 * drop_rate,
            dict(reasons),
        )

        if drop_rate >= self.drop_rate_warn:
            logger.warning(
                "High drop rate %.1f%% >= %.0f%% — inspect rejected_*.jsonl / quality_*.json",
                100.0 * drop_rate,
                100.0 * self.drop_rate_warn,
            )
        return cleaned, rejected

    def load_raw_rows(self) -> list[dict[str, Any]]:
        """Load all listings.jsonl files under raw_dir."""
        rows: list[dict[str, Any]] = []
        # sorted for reproducibility
        paths = sorted(self.raw_dir.glob("**/listings.jsonl"))
        # each snapshot file
        for path in paths:
            # read line by line
            for line in path.read_text(encoding="utf-8").splitlines():
                # skip blanks
                if not line.strip():
                    continue
                # parse JSON object
                rows.append(json.loads(line))
            logger.info("Loaded %s", path.relative_to(ROOT))
        # total
        logger.info("Raw rows loaded: %s from %s files", len(rows), len(paths))
        return rows

    

    def dedupe_latest(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:

        """Keep the latest scraped_at per (source, listing_id)."""

        best: dict[tuple[Any, Any], dict[str, Any]] = {}

        for row in rows:
            # extracting source and listing_id from row
            source = row.get("source")
            listing_id = row.get("listing_id")
            # skip rows without id
            if listing_id is None:
                continue

            # previous winner
            prev = best.get((source, listing_id))
            # take first or newer scraped_at
            if prev is None or str(row.get("scraped_at") or "") >= str(prev.get("scraped_at") or ""):
                best[(source, listing_id)] = row
        
        out = list(best.values())
        logger.info("After dedupe: %s (from %s)", len(out), len(rows))
        return out

    def check_title(self, row: dict[str, Any]) -> bool:
        """True if title looks like a whole building / non-apartment listing."""
        title = row.get("title")
        if not title:
            return False
        t = str(title).lower()
        return any(k in t for k in ("palazzo", "edificio", "intera proprietà"))
    
    @staticmethod
    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in km between two WGS84 points."""
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)
        a = (
            math.sin(d_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        return EARTH_RADIUS_KM * c

    def distance_from_center_km(self, latitude: Any, longitude: Any) -> float | None:
        """Distance from listing coords to Campidoglio; None if coords missing/invalid."""
        if latitude is None or longitude is None:
            return None
        try:
            lat = float(latitude)
            lon = float(longitude)
        except (TypeError, ValueError):
            return None
        return round(
            self.haversine_km(lat, lon, ROME_CENTER_LAT, ROME_CENTER_LON),
            3,
        )

    def clean_row(self, row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        """Clean a single row."""

        if self.check_title(row):
            return None, "non_apartment_title"

        price = row.get("price_eur_month")
        if price is None:
            return None, "missing_price"
        price = float(price)
        if price < MIN_PRICE_EUR or price > MAX_PRICE_EUR:
            return None, "price_out_of_bounds"

        rooms = row.get("rooms")
        if rooms is None:
            return None, "missing_rooms"
        match = re.match(r"(\d+)", str(rooms).strip())
        if match is None:
            return None, "missing_rooms"
        rooms = int(match.group(1))

        area = row.get("surface_m2")
        if area is None:
            return None, "missing_surface"
        match = re.search(r"(\d+(?:[.,]\d+)?)", str(area).strip())
        if match is None:
            return None, "missing_surface"
        area = float(match.group(1).strip("., "))
        if area < MIN_SURFACE_M2 or area > MAX_SURFACE_M2:
            return None, "surface_out_of_bounds"
        price_per_m2 = price / area

        row_new = {
            "source": row.get("source"),
            "listing_id": row.get("listing_id"),
            "url": row.get("url"),
            "title": row.get("title"),
            "price_eur_month": price,
            "surface_m2": area,
            "rooms": rooms,
            "bathrooms": row.get("bathrooms"),
            "floor": row.get("floor"),
            "has_elevator": row.get("has_elevator"),
            "city": row.get("city"),
            "macrozone": row.get("macrozone"),
            "microzone": row.get("microzone"),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "distance_from_center_km": self.distance_from_center_km(
                row.get("latitude"),
                row.get("longitude"),
            ),
            "contract": row.get("contract"),
            "scraped_at": row.get("scraped_at"),
            "price_per_m2_monthly": round(price_per_m2, 2),
        }
        return row_new, None


    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        """Write one JSON object per line."""
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        path.write_text(payload, encoding="utf-8")

    def write_outputs(
        self,
        cleaned: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
    ) -> Path:
        """Write cleaned, rejected quarantine, and quality summary JSON."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        out_path = self.processed_dir / f"listings_{run_id}.jsonl"
        self._write_jsonl(out_path, cleaned)
        self._write_jsonl(self.processed_dir / "listings_latest.jsonl", cleaned)

        rejected_path = self.processed_dir / f"rejected_{run_id}.jsonl"
        self._write_jsonl(rejected_path, rejected)
        self._write_jsonl(self.processed_dir / "rejected_latest.jsonl", rejected)

        quality = self.last_quality or {}
        quality_body = json.dumps(quality, ensure_ascii=False, indent=2) + "\n"
        quality_path = self.processed_dir / f"quality_{run_id}.json"
        quality_path.write_text(quality_body, encoding="utf-8")
        (self.processed_dir / "quality_latest.json").write_text(quality_body, encoding="utf-8")

        logger.info("Wrote %s cleaned → %s", len(cleaned), out_path)
        logger.info("Wrote %s rejected → %s", len(rejected), rejected_path)
        logger.info("Wrote quality report → %s", quality_path)
        return out_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and dedupe Immobiliare raw listings.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    transformer = ListingTransformer(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )
    transformer.run()


if __name__ == "__main__":
    main()
