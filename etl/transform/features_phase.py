"""Feature phase: enrich cleaned listings with RF-04 derived features."""

from __future__ import annotations

from typing import Any
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import logging
import json

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_INPUT = PROCESSED_DIR / "listings_latest.jsonl"

# Immobiliare macrozone → Roma capitale municipio (I–XV). Unmapped → None at enrich time.
MACROZONE_TO_MUNICIPIO: dict[str, str] = {
    "Centro Storico": "I",
    "Testaccio, Trastevere": "I",
    "Aventino, San Saba, Caracalla": "I",
    "Termini, Repubblica": "I",
    "Prati, Borgo, Mazzini, Delle Vittorie, Degli Eroi": "I",
    "Parioli, Flaminio": "II",
    "Salario, Trieste": "II",
    "Bologna, Policlinico": "II",
    "Talenti, Monte Sacro, Nuovo Salario": "III",
    "Porta di Roma, Casal Boccone": "III",
    "Monti Tiburtini, Pietralata": "IV",
    "Ponte Mammolo, San Basilio, Tor Cervara": "IV",
    "Pigneto, San Lorenzo, Casal Bertone": "V",
    "Alessandrino, Tor Sapienza, Torre Maura": "VI",
    "Appia Pignatelli, Ardeatino, Montagnola": "VIII",
    "Eur, Torrino, Tintoretto": "IX",
    "Cecchignola, Fonte Meravigliosa": "IX",
    "Axa, Casal Palocco, Infernetto": "X",
    "Portuense, Villa Bonelli": "XI",
    "Magliana, Trullo, Parco de' Medici": "XI",
    "Monteverde, Gianicolense, Colli Portuensi, Casaletto": "XII",
    "Casetta Mattei, Pisana, Bravetta": "XII",
    "Aurelio, Boccea": "XIII",
    "Gregorio VII, Baldo degli Ubaldi": "XIII",
    "Battistini, Torrevecchia": "XIII",
    "Trionfale, Monte Mario, Ottavia": "XIV",
    "Corso Francia, Vigna Clara, Fleming, Ponte Milvio": "XV",
    "Cassia, San Godenzo, Grottarossa": "XV",
    "Olgiata, Giustiniana": "XV",
}

# meteorological seasons by month
_SEASON_BY_MONTH = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Load cleaned listings → add features → write features JSONL."""

    def __init__(
        self,
        input_path: Path = DEFAULT_INPUT,
        processed_dir: Path = PROCESSED_DIR,
    ) -> None:
        self.input_path = input_path
        self.processed_dir = processed_dir

    def run(self) -> Path:
        rows = self.load_rows()
        enriched = self.enrich(rows)
        return self.write_outputs(enriched)

    def load_rows(self) -> list[dict[str, Any]]:
        """Load cleaned listings from a JSONL file."""
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input not found: {self.input_path}")
        rows: list[dict[str, Any]] = []
        for line in self.input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        logger.info("Loaded %s cleaned rows from %s", len(rows), self.input_path)
        return rows

    def enrich(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pass through cleaned fields; append RF-04 feature columns."""
        hist = self._leave_one_out_zone_means(rows)
        out: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            enriched = dict(row)
            enriched["area_price_per_m2_hist"] = hist[i]
            month, season = self.publication_month_season(row.get("scraped_at"))
            enriched["publication_month"] = month
            enriched["publication_season"] = season
            enriched["municipio"] = self.municipio_from_macrozone(row.get("macrozone"))
            out.append(enriched)
        logger.info("Enriched %s rows with RF-04 features", len(out))
        return out

    @staticmethod
    def publication_month_season(scraped_at: Any) -> tuple[int | None, str | None]:
        """Month (1–12) and meteorological season from scraped_at (v1 proxy for publication)."""
        if scraped_at is None or scraped_at == "":
            return None, None
        try:
            ts = datetime.fromisoformat(str(scraped_at).replace("Z", "+00:00"))
        except ValueError:
            return None, None
        month = ts.month
        return month, _SEASON_BY_MONTH[month]

    @staticmethod
    def municipio_from_macrozone(macrozone: Any) -> str | None:
        """Map Immobiliare macrozone to Roma municipio (I–XV); None if unknown/missing."""
        if macrozone is None or macrozone == "":
            return None
        return MACROZONE_TO_MUNICIPIO.get(str(macrozone))

    @staticmethod
    def _leave_one_out_zone_means(rows: list[dict[str, Any]]) -> list[float | None]:
        """Mean price_per_m2_monthly by macrozone, excluding the current row.

        Requires ≥2 rows in the same macrozone; otherwise None.
        Missing/empty macrozone → None.
        """
        sums: dict[Any, float] = defaultdict(float)
        counts: dict[Any, int] = defaultdict(int)

        for row in rows:
            zone = row.get("macrozone")
            if zone is None or zone == "":
                continue
            price = row.get("price_per_m2_monthly")
            if price is None:
                continue
            sums[zone] += float(price)
            counts[zone] += 1

        result: list[float | None] = []
        for row in rows:
            zone = row.get("macrozone")
            if zone is None or zone == "":
                result.append(None)
                continue
            n = counts.get(zone, 0)
            if n < 2:
                result.append(None)
                continue
            price = row.get("price_per_m2_monthly")
            if price is None:
                result.append(None)
                continue
            loo = (sums[zone] - float(price)) / (n - 1)
            result.append(round(loo, 2))
        return result

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        path.write_text(payload, encoding="utf-8")

    def write_outputs(self, rows: list[dict[str, Any]]) -> Path:
        """Write features_<ts>.jsonl and features_latest.jsonl."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = self.processed_dir / f"features_{run_id}.jsonl"
        self._write_jsonl(out_path, rows)
        self._write_jsonl(self.processed_dir / "features_latest.jsonl", rows)
        logger.info("Wrote %s featured rows → %s", len(rows), out_path)
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add RF-04 features to cleaned Immobiliare listings."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    FeatureBuilder(input_path=args.input, processed_dir=args.processed_dir).run()


if __name__ == "__main__":
    main()
