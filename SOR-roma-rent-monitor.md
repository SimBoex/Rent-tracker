# Statement of Requirements (SOR)
## Project: Roma Rent Monitor — ETL + ML System with Drift Monitoring

**Version**: 1.2  
**Date**: September 2026  
**Author**: [Simone Boesso]  
**Status**: v1 complete (Phases 1–4 lite); post-v1 optional: HF Spaces publish, Pandera

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
| RF-04 | The system must compute derived features (historical area €/m², distance from city center, etc.) | High | Done (lite): distance + target in clean; `area_price_per_m2_hist`, `publication_month`/`season`, `municipio` in features |
| RF-05 | The system must train a regression model to predict €/m²/month | High | Done (lite): `ml/train.py` HistGradientBoosting baseline |
| RF-06 | The system must expose an API that, given a listing, returns the predicted fair price | High | Done (lite): `api/` FastAPI + `models/baseline_latest` |
| RF-07 | The system must compute the gap between actual and predicted price and classify "good deal / fair price / above market" | Medium | Done (lite): `/predict` gap_pct + deal_label (±10% band) |
| RF-08 | The system must generate periodic data-drift and prediction-drift reports | High | Done (lite): `ml/drift_report.py` Evidently HTML + summary |
| RF-09 | The system must trigger automatic retraining when drift/error exceeds a defined threshold | Medium | Done (lite): `ml/retrain_check.py` — MAE ratio ≥ 1.5 and `n_reference` ≥ 50 |
| RF-10 | The system must show a dashboard with monitoring metrics and listings flagged as "good deal" | Medium | Done (lite): local Streamlit `dashboard/app.py`; public try-predict = Gradio HF Space → Render (`doc/hf_spaces.md`) |
| RF-11 | The system must track every training experiment (parameters, metrics, model version) | High | Done (lite): local MLflow SQLite (`mlflow.db`) |
| RF-12 | The system must version the datasets used for each training run | Medium | Done (lite): SHA-256 fingerprint → `dataset.json` + MLflow `dataset_sha256` (DVC remote later) |

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
[Transform clean: ListingTransformer]
        → dedupe (source, listing_id) keep latest scraped_at
        → clean / quality gate
        → data/processed/listings_*.jsonl
        → data/processed/rejected_*.jsonl   (quarantine + drop_reason)
        → data/processed/quality_*.json     (counts, drop_rate, by_reason)
        ↓
[Transform features: FeatureBuilder]
        → RF-04 derived features (e.g. historical area €/m²)
        → data/processed/features_*.jsonl
        ↓
[Load: DB]  (future)
        ↓
[Training: ml/train.py + MLflow local] → [Serving API (FastAPI: api/main.py)]
        ↓
[Monitoring: Evidently drift + retrain] → [Dashboard Streamlit local / Gradio HF]
```

### Current modules
| Stage | Module | Role |
|---|---|---|
| Extract | `etl/extract/immobiliare_scraper.py` | `ImmobiliareScraper`: robots.txt, polite delay, `__NEXT_DATA__` parse |
| Transform clean | `python -m etl.transform.clean_phase` | `ListingTransformer`: clean, dedupe, target, quality monitoring |
| Transform features | `python -m etl.transform.features_phase` | `FeatureBuilder`: RF-04 features on cleaned listings → `features_*.jsonl` |
| Training | `python -m ml.train` | Baseline `HistGradientBoostingRegressor`; metrics + `dataset.json` (RF-12) + local MLflow |
| Drift | `python -m ml.drift_report` | Evidently `DataDriftPreset` on features (+ prediction if model present) → `reports/` |
| Retrain gate | `python -m ml.retrain_check` | RF-09: retrain if `mae_current/mae_reference` ≥ 1.5 and `n_reference` ≥ 50 → `reports/retrain_*/decision.json` |
| Dashboard | Streamlit local / Gradio on HF | RF-10 local + public try-predict via Render (`dashboard/gradio_app.py`, `doc/hf_spaces.md`) |
| Serving | `uvicorn api.main:app` / `Dockerfile` | Load `models/baseline_latest`; `POST /predict`, `GET /health` |
| Orchestration (local) | `run_pipeline.py` | Optional scrape → clean → features → train |
| Orchestration (CI) | `.github/workflows/daily_monitoring.yml` | Daily scrape(~100 pages, no HTML) → features (merge history) → prune raw → drift → retrain-if-MAE-gate |

### Proposed tech stack
| Component | Technology |
|---|---|
| Scraping | Python (curl_cffi + BeautifulSoup) |
| Transform / quality gate (v1) | stdlib JSONL + drop-reason counters; Pandera later |
| Orchestration | GitHub Actions (v1) → Airflow (future) |
| Storage | JSONL files (v1) → SQLite → Postgres/S3 (future) |
| Data versioning | Content hash per train run (`dataset.json`); DVC + R2/S3 remote optional — see `doc/dvc.md` |
| Training/tracking | scikit-learn / XGBoost + MLflow |
| Serving | FastAPI + Docker |
| Deployment | Render/Railway or AWS Lambda |
| Drift monitoring | Evidently AI (market/model drift; separate from transform quality gate) |
| Dashboard | Streamlit (local) + Gradio (HF Spaces → Render API) |

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

### 5.5 Feature layer (`features_*.jsonl`)
- `area_price_per_m2_hist` — leave-one-out mean of `price_per_m2_monthly` by `macrozone` (≥2 rows in zone; else null)
- `publication_month` / `publication_season` — from `scraped_at` (v1 proxy until a true publication date is scraped)
- `municipio` — Immobiliare `macrozone` → Roma capitale municipio I–XV (`MACROZONE_TO_MUNICIPIO` in `features_phase.py`; unknown → null)

### 5.6 Target
`price_per_m2_monthly` (monthly rent / surface in m²)

### 5.7 Quality gate (transform)
Residential bounds (configurable): surface 15–400 m²; price €200–€15 000/month.  
Drop reasons tracked: `missing_surface`, `missing_price`, `missing_rooms`, `surface_out_of_bounds`, `price_out_of_bounds`, `non_apartment_title`.  
Alert when clean-stage `drop_rate` ≥ 25% (likely parser/gate bug — inspect `rejected_*.jsonl`).

### 5.8 Data constraints
- Train/test split must be **temporal**, not random: last scrape day = test when ≥2 distinct `scraped_at` days; with a single day, `ordered_holdout_fallback` (last 20% by `(scraped_at, listing_id)`) until more scrapes exist
- Zone normalization: `macrozone` → `municipio` in feature phase (`MACROZONE_TO_MUNICIPIO`)
- Do not commit raw/processed listings to the public repo

---

## 6. Success Metrics

| Metric | Indicative target |
|---|---|
| Model MAE on temporal test set | To define after initial baseline |
| Data coverage | At least 3–5 Rome zones with enough data after 4–6 weeks of scraping |
| Transform quality monitoring | Every transform run writes `quality_*.json` + quarantine |
| Active drift monitoring | Report generated automatically on a daily cadence (Phase 4) |
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

**v1 closed** — all High/Medium functional requirements (RF-01–RF-12) delivered in lite form.

1. **Phase 1**: Scraper + first raw data — **done**
2. **Phase 2**: Full ETL pipeline + first model version — **done** (ETL + baseline train + local API)
3. **Phase 3**: API deployment — **done** (Docker + Render)
4. **Phase 4**: Monitoring, drift, retrain, dashboard, daily GHA cadence — **done (lite)** (HF Spaces: wire to Render via `RENT_API_URL`)

**Post-v1 (optional):** Publish HF Space (see `doc/hf_spaces.md`), Pandera quality schemas, second source (Idealista), economic validation of “good deal”.

---

## 9. Open questions / To decide

- [ ] Exact number and names of Rome zones to emphasize in the portfolio narrative
- [x] Exact drift/error threshold that triggers retraining — default **MAE ratio ≥ 1.5** and **`n_reference` ≥ 50** (`ml.retrain_check`; CLI overrides)
- [x] Exact scraping frequency (daily vs weekly) — default **daily** (GitHub Actions 06:00 UTC; `workflow_dispatch` for manual)
- [ ] Whether to add a second data source (Idealista) in a later iteration
- [ ] How to validate that “good deal” listings (large gap: actual rent ≪ predicted fair €/m²) are actually rented faster (e.g. shorter time-on-market / disappear sooner from scrape snapshots) — needed to prove RF-07 is economically useful, not just a model residual
- [x] Transform drop-rate warn threshold — default **25%** (`DROP_RATE_WARN`)
- [x] Dedup strategy — raw snapshots immutable; upsert on `(source, listing_id)` in transform
