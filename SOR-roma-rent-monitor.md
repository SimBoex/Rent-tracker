# Statement of Requirements (SOR)
## Project: Roma Rent Monitor — ETL + ML System with Drift Monitoring

**Version**: 1.0  
**Date**: September 2026  
**Author**: [Name]  
**Status**: Initial draft  

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

---

## 2. Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| RF-01 | The system must periodically extract rental listings from at least one platform (Immobiliare.it) | High |
| RF-02 | The system must normalize and clean raw data (areas, missing values, duplicates) | High |
| RF-03 | The system must validate the data schema before loading (data quality gate) | High |
| RF-04 | The system must compute derived features (historical area €/m², distance from city center, etc.) | High |
| RF-05 | The system must train a regression model to predict €/m²/month | High |
| RF-06 | The system must expose an API that, given a listing, returns the predicted fair price | High |
| RF-07 | The system must compute the gap between actual and predicted price and classify "good deal / fair price / above market" | Medium |
| RF-08 | The system must generate periodic data-drift and prediction-drift reports | High |
| RF-09 | The system must trigger automatic retraining when drift/error exceeds a defined threshold | Medium |
| RF-10 | The system must show a dashboard with monitoring metrics and listings flagged as "good deal" | Medium |
| RF-11 | The system must track every training experiment (parameters, metrics, model version) | High |
| RF-12 | The system must version the datasets used for each training run | Medium |

---

## 3. Non-Functional Requirements

| ID | Requirement | Notes |
|----|-------------|-------|
| RNF-01 | Near-zero infrastructure cost | Use free tiers (Render/Railway, AWS free tier, HF Spaces) |
| RNF-02 | Polite scraping: rate limiting, respect robots.txt | 1 request every 3–5 seconds |
| RNF-03 | No public redistribution of raw scraped data | Repo only: aggregated data / synthetic sample |
| RNF-04 | Reproducible pipeline (no notebooks in production, versioned scripts only) | |
| RNF-05 | Tested code (at least unit tests on transform and feature engineering) | |
| RNF-06 | Complete technical documentation (README, architecture, motivated design decisions) | Required for portfolio use |
| RNF-07 | Temporal train/test split (never random) to avoid data leakage | |

---

## 4. High-Level Architecture

```
[Extract: Scraper] → [Transform: clean/validate/feature eng.] → [Load: DB]
                                                                      ↓
                                                          [Training pipeline + MLflow]
                                                                      ↓
                                                          [Serving API (FastAPI)]
                                                                      ↓
                                            [Monitoring: drift report + retrain trigger]
                                                                      ↓
                                                          [Dashboard (Streamlit)]
```

### Proposed tech stack
| Component | Technology |
|---|---|
| Scraping | Python (curl_cffi + BeautifulSoup) |
| Orchestration | GitHub Actions (v1) → Airflow (future) |
| Data validation | Pandera |
| Storage | SQLite (v1) → Postgres/S3 (future) |
| Data versioning | DVC |
| Training/tracking | scikit-learn / XGBoost + MLflow |
| Serving | FastAPI + Docker |
| Deployment | Render/Railway or AWS Lambda |
| Drift monitoring | Evidently AI |
| Dashboard | Streamlit (Hugging Face Spaces) |

---

## 5. Data

### 5.1 Source
Immobiliare.it — rentals section, Rome area

### 5.2 Planned features (v1)
Surface area, number of rooms, floor, elevator, zone/municipality, distance from city center, publication month/season, historical area €/m².

### 5.3 Target
`price_per_m2_monthly` (monthly rent / surface in m²)

### 5.4 Data constraints
- Train/test split must be **temporal**, not random
- Zone normalization handled centrally (zone → municipality map)

---

## 6. Success Metrics

| Metric | Indicative target |
|---|---|
| Model MAE on temporal test set | To define after initial baseline |
| Data coverage | At least 3–5 Rome zones with enough data after 4–6 weeks of scraping |
| Active drift monitoring | Report generated automatically on a weekly cadence |
| Working end-to-end pipeline | From extract to dashboard, with no manual intervention |

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Scraping site changes HTML structure | Parser isolated in a dedicated, easy-to-update module |
| Insufficient data in the first weeks | Start training/pipeline even with little data; enrich over time |
| IP block from overly aggressive scraping | Strict rate limiting; user-agent rotation if needed |
| "False" drift from pipeline bugs rather than the market | Upstream data validation (RF-03) to separate technical errors from real drift |

---

## 8. Roadmap (reference)

1. **Phase 1**: Scraper + first raw data
2. **Phase 2**: Full ETL pipeline + first model version
3. **Phase 3**: API deployment
4. **Phase 4**: Monitoring, drift report, automatic retraining, dashboard

---

## 9. Open questions / To decide

- [ ] Exact number and names of Rome zones to include in v1
- [ ] Exact drift/error threshold that triggers retraining
- [ ] Exact scraping frequency (daily vs weekly)
- [ ] Whether to add a second data source (Idealista) already in v1 or in a later iteration
