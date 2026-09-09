"""

Extract: scrape listing pages for Rome rentals from Immobiliare.it.

"""

# to postpone the evaluation of the type hints
from __future__ import annotations

import argparse
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse
# to parse the robots.txt file
from urllib.robotparser import RobotFileParser
# to make the HTTP requests avoiding the anti-bot protection
from curl_cffi import requests as cffi_requests
# to parse the HTML content
from bs4 import BeautifulSoup

import json

BASE_URL = "https://www.immobiliare.it"
# Italian list URL for Rome rentals (allowed by robots.txt; avoid /search-list, /ricerca-mappa).
DEFAULT_START_URL = f"{BASE_URL}/affitto-case/roma/"
# Used only for robots.txt matching (site rules are under User-agent: *).
ROBOTS_UA = "Mozilla/5.0"
# TLS fingerprint: plain requests gets 403 from anti-bot; curl_cffi Chrome works.
IMPERSONATE = "chrome"
MIN_DELAY_S = 3.0
MAX_DELAY_S = 5.0
REQUEST_TIMEOUT_S = 30

# to get the root directory of the project
ROOT = Path(__file__).resolve().parents[2]
# to save the raw HTML files
RAW_DIR = ROOT / "data" / "raw"

# to log the messages
logger = logging.getLogger(__name__)




class RobotsChecker:
    """ the responsability of this class is to check if the URL is allowed by the robots.txt file """

    def __init__(self, base_url: str, session: cffi_requests.Session, user_agent: str = ROBOTS_UA) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.user_agent = user_agent
        self._parser: RobotFileParser | None = None


    def _ensure_loaded(self) -> RobotFileParser:
        if self._parser is None:
            robots_url = urljoin(self.base_url, "/robots.txt")
            resp = self.session.get(robots_url, timeout=REQUEST_TIMEOUT_S)
            resp.raise_for_status()
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.parse(resp.text.splitlines())
            self._parser = rp
            logger.debug("Loaded robots.txt (%s bytes)", len(resp.text))
        return self._parser
    
    # it downloads the robots.txt file only at the first call
    def allowed(self, url: str) -> bool:
        return self._ensure_loaded().can_fetch(self.user_agent, url)


def polite_get(session: cffi_requests.Session, url: str, robots: RobotsChecker) -> cffi_requests.Response:
    if not robots.allowed(url):
        raise PermissionError(f"robots.txt disallows fetching: {url}")
    
    time.sleep(random.uniform(MIN_DELAY_S, MAX_DELAY_S))
    resp = session.get(url, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp

def extract_next_data(html: str) -> dict[str, Any] | None:
    """Immobiliare (Next.js framework) mette i listing in <script id="__NEXT_DATA__">."""
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        return None
    return json.loads(tag.string)

# _NEXT_DATA__  is an enourmous tree of data, so we need to search for the results[] list recursively
def _find_results(node: Any) -> list[Any] | None:
    """recursively search for the results[] list in the JSON"""
    if isinstance(node, dict):
        results = node.get("results")
        # we are looking for a list of dictionaries!
        if isinstance(results, list) and results and isinstance(results[0], dict):
            sample = results[0]
            if "realEstate" in sample or "id" in sample or "price" in sample:
                return results
        for value in node.values():
            found = _find_results(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_results(item)
            if found is not None:
                return found
    return None


# to avoid crush on missing keys, we use a default value
def _dig(obj: Any, *Keys: str, default: Any = None) -> Any:
    """recursively search for the key in the object"""
    cur = obj
    for key in Keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

# to extract a row for each item in the results[] list (see row 99)
def normalize_listing(item: dict[str, Any], scraped_at: str) -> dict[str, Any]:
    """normalzing an element of results[] list"""
    
    re_obj = item.get("realEstate") if isinstance(item.get("realEstate"), dict) else item
    # list of details of the property
    props_list = re_obj.get("properties") or []
    prop = props_list[0] if props_list and isinstance(props_list[0], dict) else {}
    
    location = prop.get("location") if isinstance(prop.get("location"), dict) else {}
    price_raw = re_obj.get("price")
    price = price_raw.get("value") if isinstance(price_raw, dict) else price_raw
    
    floor_raw = prop.get("floor")
    if isinstance(floor_raw, dict):
        # take the first non-empty value
        floor = (
            floor_raw.get("abbreviation")
            or floor_raw.get("value")
            or floor_raw.get("floorOnlyValue")
        )
    else:
        floor = floor_raw
    

    # URL is on item.seo (sibling of realEstate), not on realEstate.seo
    item_seo = item.get("seo") if isinstance(item.get("seo"), dict) else {}
    # keep realEstate.seo as fallback if payload shape changes
    re_seo = re_obj.get("seo") if isinstance(re_obj.get("seo"), dict) else {}
    # prefer absolute listing URL from item.seo.url (e.g. /annunci/<id>/)
    path = (
        item_seo.get("url")
        or re_seo.get("url")
        or _dig(re_obj, "urls", "default")
        or ""
    )
    # urljoin keeps absolute https URLs unchanged; joins relative paths to BASE_URL
    url = urljoin(BASE_URL, path) if path else None
    
    has_elevator = None
    elevators = prop.get("elevators")
    if elevators is not None:
        has_elevator = bool(elevators) if not isinstance(elevators, bool) else elevators
    elif "elevator" in prop: # alternative way to check if the property has an elevator
        has_elevator = bool(prop.get("elevator"))
    
    return {
        "source": "immobiliare.it",
        "listing_id": re_obj.get("id") or item.get("id"),
        "url": url,
        "title": re_obj.get("title") or prop.get("caption"),
        "price_eur_month": price,
        "surface_m2": prop.get("surface") or prop.get("surfaceValue"),
        "rooms": prop.get("rooms") or prop.get("roomsCount"),
        "bathrooms": prop.get("bathrooms"),
        "floor": floor,
        "has_elevator": has_elevator,
        "city": location.get("city"),
        "macrozone": location.get("macrozone") or location.get("macroZone"),
        "microzone": location.get("microzone") or location.get("microZone"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "contract": re_obj.get("contract"),
        "scraped_at": scraped_at,
    }
    
def parse_listings_from_html(html: str, scraped_at: str) -> list[dict[str, Any]]:
    next_data = extract_next_data(html)
    if not next_data:
        logger.warning("No __NEXT_DATA__ — layout changed or bot block.")
        return []
    results = _find_results(next_data)
    if not results:
        logger.warning("results[] not found inside __NEXT_DATA__.")
        return []
    return [
        normalize_listing(item, scraped_at)
        for item in results
        if isinstance(item, dict)
    ]


def page_url(start_url: str, page: int) -> str:
    """generate the URL for the next page"""
    if page <= 1:
        return start_url
    parsed = urlparse(start_url)
    # query string is the part of the URL after the ? or empty string
    # it returns a dictionary of key-value pairs   (& is the delimiter)
    query = parse_qs(parsed.query)
    # overwrite the page parameter with the new page number
    query["pag"] = [str(page)]
    # transform back each value to a string
    flat = {k: v[0] for k, v in query.items()}
    return urlunparse(parsed._replace(query=urlencode(flat)))    


class ImmobiliareScraper:
    """Orchestrazione: session + robots + loop pagine + salvataggio."""
    def __init__(
        self,
        start_url: str = DEFAULT_START_URL,
        save_html: bool = True,
    ) -> None:
        self.start_url = start_url
        self.save_html = save_html
        self.session = cffi_requests.Session(impersonate=IMPERSONATE)
        self.robots = RobotsChecker(BASE_URL, self.session)

    def run(self, max_pages: int = 1, start_page: int = 1) -> Path:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = RAW_DIR / f"immobiliare_roma_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Design Choice: .jsonl is a good format for streaming data (one JSON object per line)
        out_jsonl = out_dir / "listings.jsonl"
        scraped_at = datetime.now(timezone.utc).isoformat()
        total = 0
        start_page = max(1, start_page)
        end_page = start_page + max(1, max_pages) - 1
        with out_jsonl.open("w", encoding="utf-8") as fh:
            for page in range(start_page, end_page + 1):
                url = page_url(self.start_url, page)
                logger.info("Fetching page %s: %s", page, url)
                try:
                    resp = polite_get(self.session, url, self.robots)
                except PermissionError:
                    # fail hard on robots.txt violation
                    raise
                except Exception as exc:
                    # fail gracefully on other errors
                    logger.error("HTTP error on %s: %s", url, exc)
                    break

                html = resp.text
                # in order to debug the scraper, we save the HTML files
                if self.save_html:
                    (out_dir / f"page_{page:03d}.html").write_text(html, encoding="utf-8")
                listings = parse_listings_from_html(html, scraped_at)
                if not listings:
                    logger.warning("No listings on page %s — stopping.", page)
                    break
                for row in listings:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                total += len(listings)
                logger.info("Page %s: %s listings", page, len(listings))
        logger.info("Done. %s listings → %s", total, out_jsonl)
        return out_jsonl



def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Immobiliare.it Rome rentals.")
    parser.add_argument("--start-url", default=DEFAULT_START_URL)
    parser.add_argument("--start-page", type=int, default=1, help="First page to fetch (default 1)")
    parser.add_argument("--max-pages", type=int, default=1, help="How many pages to fetch from start-page")
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    scraper = ImmobiliareScraper(
        start_url=args.start_url,
        save_html=not args.no_html,
    )
    scraper.run(max_pages=args.max_pages, start_page=args.start_page)


if __name__ == "__main__":
    main()


