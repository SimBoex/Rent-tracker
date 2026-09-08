# Statement of Requirements (SOR)
## Project: Roma Rent Monitor — ETL + ML System with Drift Monitoring

**Version**: 1.1  
**Date**: September 2026  
**Author**: [Name]  
**Status**: In progress (Phase 1 done; Phase 2 transform in progress)  

---

## 1. Context and Goal

### 1.1 Context
Personal portfolio project aimed at demonstrating end-to-end skills in:
- Data engineering (ETL pipeline)
- Machine Learning (regression model)
- MLOps (deployment, monitoring, automatic retraining)

### 1.2 Primary goal
Build a system that:
1. Periodically collects rental listings in Rome
2. Predicts fair rent value (€/m²/month) by area and property features
3. Flags listings that are above/below market
4. Monitors data and performance drift over time, with automatic retraining

### 1.3 Secondary goal (portfolio)
Show concrete, verifiable interview-ready experience with: orchestrated ETL pipelines, experiment tracking, containerization, cloud deployment, data/concept drift monitoring, CI/CD.

### 1.4 Out of scope (v1)
- Property sales market (rentals only)
- Analysis from listing images/photos
- Mobile app or complex UI (basic dashboard only)
- Multi-city coverage (Rome only)
- Multiple data sources in v1 (start from one scraping platform; architecture designed to extend later)
- Public redistribution of raw scraped listings (portfolio shows code, metrics, and model — not raw dumps)

---

## 2. Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| RF-01 | The system must periodically extract rental listings from at least one platform (Immobiliare.it) | High | Done (scraper) |
| RF-02 | The system must normalize and clean raw data (areas, missing values, duplicates) | High | Done (transform) |
| RF-03 | The system must validate listings with a data quality gate before they enter processed/training data | High | Done (lite); Pandera schema later |
| RF-03b | The system must record drop reasons, quarantine rejected rows, and emit a per-run quality report (to catch pipeline bugs vs market drift) | High | Done |
| RF-04 | The system must compute derived features (historical area €/m², distance from city center, etc.) | High | Partial (`price_per_m2_monthly`, `distance_from_center_km`; historical area €/m² still todo) |
| RF-05 | The system must train a regression model to predict €/m²/month | High | Todo |
| RF-06 | The system must expose an API that, given a listing, returns the predicted fair price | High | Todo |
| RF-07 | The system must compute the gap between actual and predicted price and classify "good deal / fair price / above market" | Medium | Todo |
| RF-08 | The system must generate periodic data-drift and prediction-drift reports | High | Todo (Evidently) |
| RF-09 | The system must trigger automatic retraining when drift/error exceeds a defined threshold | Medium | Todo |
| RF-10 | The system must show a dashboard with monitoring metrics and listings flagged as "good deal" | Medium | Todo |
| RF-11 | The system must track every training experiment (parameters, metrics, model version) | High | Todo |
| RF-12 | The system must version the datasets used for each training run | Medium | Todo |

---

## 3. Non-Functional Requirements

| ID | Requirement | Notes |
|----|-------------|-------|
| RNF-01 | Near-zero infrastructure cost | Use free tiers (Render/Railway, AWS free tier, HF Spaces) |
| RNF-02 | Polite scraping: rate limiting, respect robots.txt | 1 request every 3–5 seconds |
| RNF-03 | No public redistribution of raw scraped data | Repo only: aggregated data / synthetic sample; `data/raw` and `data/processed` gitignored |
| RNF-04 | Reproducible pipeline (no notebooks in production, versioned scripts only) | |
| RNF-05 | Tested code (at least unit tests on transform and feature engineering) | |
| RNF-06 | Complete technical documentation (README, architecture, motivated design decisions) | Required for portfolio use |
| RNF-07 | Temporal train/test split (never random) to avoid data leakage | |

---

## 4. High-Level Architecture

```
[Extract: ImmobiliareScraper]
        → data/raw/immobiliare_roma_<ts>/listings.jsonl  (immutable snapshots)
        ↓
[Transform: ListingsTransformer]
        → dedupe (source, listing_id) keep latest scraped_at
        → clean / quality gate
        → data/processed/listings_*.jsonl
        → data/processed/rejected_*.jsonl   (quarantine + drop_reason)
        → data/processed/quality_*.json     (counts, drop_rate, by_reason)
        ↓
[Load: DB]  (future)
        ↓
[Training pipeline + MLflow] → [Serving API (FastAPI)]
        ↓
[Monitoring: Evidently drift + retrain] → [Dashboard (Streamlit)]
```

### Current modules
| Stage | Module | Role |
|---|---|---|
| Extract | `etl/extract/immobiliare_scraper.py` | `ImmobiliareScraper`: robots.txt, polite delay, `__NEXT_DATA__` parse |
| Transform | `etl/transform/clean_phase.py` | `ListingTransformer`: clean, dedupe, target, quality monitoring |

### Proposed tech stack
| Component | Technology |
|---|---|
| Scraping | Python (curl_cffi + BeautifulSoup) |
| Transform / quality gate (v1) | stdlib JSONL + drop-reason counters; Pandera later |
| Orchestration | GitHub Actions (v1) → Airflow (future) |
| Storage | JSONL files (v1) → SQLite → Postgres/S3 (future) |
| Data versioning | DVC |
| Training/tracking | scikit-learn / XGBoost + MLflow |
| Serving | FastAPI + Docker |
| Deployment | Render/Railway or AWS Lambda |
| Drift monitoring | Evidently AI (market/model drift; separate from transform quality gate) |
| Dashboard | Streamlit (Hugging Face Spaces) |

---

## 5. Data

### 5.1 Source
Immobiliare.it — rentals section, Rome area (`/affitto-case/roma/`)

### 5.2 Raw layer
- Append-only snapshots per scrape run under `data/raw/`
- Cross-run duplicates by `listing_id` are expected; dedupe happens in transform

### 5.3 Processed fields (v1 clean)
`source`, `listing_id`, `url`, `title`, `price_eur_month`, `surface_m2`, `rooms`, `bathrooms`, `floor`, `has_elevator`, `city`, `macrozone`, `microzone`, `latitude`, `longitude`, `distance_from_center_km`, `contract`, `scraped_at`, `price_per_m2_monthly`

### 5.4 Distance from center
Haversine km from listing coords to Piazza del Campidoglio (`41.8934`, `12.4829`). Missing/invalid coords → `null` (row not dropped).

### 5.5 Planned extra features (still todo)
Publication month/season, historical area €/m², zone → municipality map

### 5.6 Target
`price_per_m2_monthly` (monthly rent / surface in m²)

### 5.7 Quality gate (transform)
Residential bounds (configurable): surface 15–400 m²; price €200–€15 000/month.  
Drop reasons tracked: `missing_surface`, `missing_price`, `missing_rooms`, `surface_out_of_bounds`, `price_out_of_bounds`, `non_apartment_title`.  
Alert when clean-stage `drop_rate` ≥ 25% (likely parser/gate bug — inspect `rejected_*.jsonl`).

### 5.8 Data constraints
- Train/test split must be **temporal**, not random
- Zone normalization handled centrally (zone → municipality map) — todo
- Do not commit raw/processed listings to the public repo

---

## 6. Success Metrics

| Metric | Indicative target |
|---|---|
| Model MAE on temporal test set | To define after initial baseline |
| Data coverage | At least 3–5 Rome zones with enough data after 4–6 weeks of scraping |
| Transform quality monitoring | Every transform run writes `quality_*.json` + quarantine |
| Active drift monitoring | Report generated automatically on a weekly cadence (Phase 4) |
| Working end-to-end pipeline | From extract to dashboard, with no manual intervention |

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Scraping site changes HTML / `__NEXT_DATA__` shape | Parser isolated in extract module; transform quality drop-rate alert |
| Insufficient data in the first weeks | Start training/pipeline even with little data; enrich over time |
| IP block from overly aggressive scraping | Strict rate limiting; respect robots.txt |
| "False" drift from pipeline bugs rather than the market | Transform quality gate + rejected quarantine + `by_reason` report **before** Evidently drift (RF-03b) |
| Accidental publication of scraped raw data | `.gitignore` on `data/raw` and `data/processed`; portfolio publishes model/metrics only |

---

## 8. Roadmap (reference)

1. **Phase 1**: Scraper + first raw data — **done**
2. **Phase 2**: Full ETL pipeline + first model version — **in progress** (clean/dedupe/quality done; features + model next)
3. **Phase 3**: API deployment
4. **Phase 4**: Monitoring, drift report, automatic retraining, dashboard

---

## 9. Open questions / To decide

- [ ] Exact number and names of Rome zones to include in v1
- [ ] Exact drift/error threshold that triggers retraining
- [ ] Exact scraping frequency (daily vs weekly)
- [ ] Whether to add a second data source (Idealista) already in v1 or in a later iteration
- [x] Transform drop-rate warn threshold — default **25%** (`DROP_RATE_WARN`)
- [x] Dedup strategy — raw snapshots immutable; upsert on `(source, listing_id)` in transform
