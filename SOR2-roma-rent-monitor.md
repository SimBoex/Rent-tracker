# Statement of Requirements (SOR) — v2
## Project: Roma Rent Monitor — ETL + ML System with Drift Monitoring

**Version**: 2.0  
**Date**: September 2026  
**Author**: [Simone Boesso]  
**Status**: Active — **OMI Open Data** (Agenzia Entrate). Immobiliare.it scrape **removed from code**; see [`SOR-roma-rent-monitor.md`](SOR-roma-rent-monitor.md) for v1 history.  
**Source of truth for ingest:** [`doc/omi.md`](doc/omi.md).

---

## 1. Context

### 1.1 Why v2
v1 (Immobiliare scrape) was portfolio-complete but had known lite shortcuts and ToS / redistribution friction. v2 switches primary extract to official **OMI** quotazioni (zone-level €/m²).

### 1.2 Goals
1. Ingest official **OMI** locazione quotazioni for Roma (semester cadence; manual download + DVC/CI)
2. Predict fair rent €/m²/month from zone / typology / conservation (+ geo later)
3. Flag zones **below/above model** (and later OMI band) — not listing “good deals”
4. Monitor drift / retrain; serve API + dashboard — near-zero cost; cite «Agenzia Entrate – OMI»

### 1.3 Out of scope
- Sales market, mobile app, multi-city (unless promoted later)
- Full listing-image / vision models (optional stretch only)
- Public redistribution of raw OMI dumps beyond attributed aggregates (RNF-03)
- Immobiliare.it (or other portal) scraping

### 1.4 Data-source note
**OMI ≠ individual ads.** Zone-level min/max by zona / tipologia / stato. Product is zone fair rent + band/model flags (RF-10b / RF-07e).

---

## 2. Functional requirements (must)

| ID | Requirement | Priority | Notes / acceptance |
|----|-------------|----------|-------------------|
| **RF-01d** | Primary source = **OMI Open Data**. | High | Quotazioni locazione + zone perimeters. See `doc/omi.md`. Manual download; cloud via DVC. |
| **RF-10b** | Public “deals” = **zone-level** vs model / OMI band | High | No listing URLs. |
| **RF-04b** | **`municipio` / zona from coordinates** (point-in-polygon) | High | Map click (RF-06c) → lat/lon → zona OMI. |
| **RF-05b** | **Model selection / HPO** with **temporal** splits by **semester** | High | MLflow; promote best to `baseline_latest`. |
| **RF-07e** | Deal label: `below_omi_band` / `in_band` / `above_omi_band` (vs model and/or band) | Medium | Public UI zone table. |

### 2.1 Clarifications

**RF-10b** — Public UI shows zone/tipologia rows vs model residual (and later OMI band).  

**RF-04b** — Store boundary source + version; unit tests on known points; align with RF-06c.

**RF-05b** — No random K-fold on shuffled rows; temporal / semester splits.

### 2.2 OMI engineering checklist

1. Download OMI quotazioni (+ perimetri) for Roma → `data/raw/omi/` (gitignored; DVC).  
2. `etl/extract/omi_loader.py` + `etl/transform/omi_features.py` → `features_latest.jsonl`.  
3. Target = mid `(LOCMIN+LOCMAX)/2`; no current LOCMIN/LOCMAX in features.  
4. Train/API/UI on OMI schema; CI = `workflow_dispatch` + optional schedule after DVC push.  
5. Cite «Agenzia Entrate – OMI» in docs and UI footer.

---

## 3. Further v2 improvements (recommended backlog)

Prioritized for portfolio impact vs effort. Not all required to “close” v2; pick a thin vertical slice.

### 3.1 Data quality & semantics (High value)

| ID | Improvement | Why |
|----|-------------|-----|
| RF-03c | **Pandera** (or equivalent) schemas on clean/features | Was deferred in v1; catches parser breaks before drift false alarms |
| RF-04c | Prefer true **semester / publication** fields over synthetic `scraped_at` when present in export | Better temporal split |
| RF-07b | Validate zone flags against semester-over-semester OMI shifts | Economic usefulness of RF-07e |
| **RF-07c** | **Condition / quality proxies** so visually or structurally “bad” units are less likely to be labeled good deals **without** using photos | Extend title/description heuristics beyond `non_apartment_title` (e.g. *da ristrutturare*, *da rifare*, *asta*, *nuda proprietà*, *occupazione*); optional completeness gates (missing floor/elevator, extreme price outliers). Apply as **filter or soft penalty** on good-deal export and/or `/predict` deal_label. Document that good deal = under market **on observed features**, not “nice flat”. |
| RF-07d | *(stretch)* Listing **photo** condition score (vision embedding / classifier) as feature or post-filter | Only if portfolio wants multimodal; otherwise keep out of scope |
| RF-01b | Optional **second source** (e.g. Idealista) behind the same clean schema | Robustness + portfolio “multi-source ETL” |

### 3.2 Modeling & MLOps (High value)

| ID | Improvement | Why |
|----|-------------|-----|
| RF-05c | Prediction **intervals** or quantile regression | Honest uncertainty on `/predict` and UI |
| RF-09b | Retrain gate considers **data drift + MAE** (documented policy) | Aligns RF-08 and RF-09 beyond MAE-only lite |
| RF-12b | Promote model artifact via **DVC/R2** (or model registry) so Render API picks new weights without relying only on git-lfs-ish commits | Cleaner deploy after CI retrain |
| RF-11b | Keep MLflow UI export or remote tracking for interview demos | Show experiment comparison from RF-05b |

### 3.3 Product / UI (Medium)

| ID | Improvement | Why |
|----|-------------|-----|
| RF-10c | Public UI: Predict + Monitoring + Good deals (current-only, anonymized) — already directionally started | Keep RNF-03 |
| RF-10d | Optional SHAP / top feature contributions on Predict | Interview-friendly explainability |
| **RF-06c** | Predict UI: **`distance_from_center_km` (and lat/lon) via map click** — user picks a point on a Rome map instead of typing the distance by hand | Improves UX and feeds RF-04b (municipio from coordinates). API may still accept numeric distance; UI derives km (Haversine to Campidoglio or chosen reference) + optional lat/lon from the click. Fallback: manual number input. |
| RF-06b | Batch `/predict` or score-all-zones endpoint for internal zone-deal export | Simplifies RF-10b pipeline |

### 3.4 Performance / parallelism

| ID | Improvement | Why |
|----|-------------|-----|
| **RF-04d** | **Parallelize CPU-bound feature / geo** where safe (point-in-polygon, chunked enrichment) | Larger multi-semester OMI + boundaries; keep deterministic for RF-12 |

### 3.5 Reliability

| ID | Improvement | Why |
|----|-------------|-----|
| RNF-05c | Contract tests on OMI CSV column aliases / fixture shape | Catch export format changes |
| RNF-01b | Keep free-tier: Render UI + API | Cost |

---

## 4. Non-functional (carry-over + v2 notes)

| ID | Requirement | v2 note |
|----|-------------|---------|
| RNF-01 | Near-zero cost | Unchanged |
| RNF-02 | *(v1 scrape politeness — N/A)* | No portal scrape in v2 |
| RNF-03 | No public raw redistribution | Zone aggregates / anonymized snapshots only |
| RNF-04 | Reproducible scripts | HPO config versioned (YAML/CLI) |
| RNF-05 | Tests | Municipio geo fixtures; HPO smoke; OMI loader fixtures |
| RNF-06 | Documentation | `doc/omi.md`, `doc/usage.md`, `doc/render.md` |
| RNF-07 | Temporal splits | Mandatory for RF-05b |

---

## 5. Suggested v2 roadmap

1. **OMI ingest (done):** loader + features + train on fixture / real semester (`doc/omi.md`)  
2. **Geography:** point → zona OMI (+ municipio) (RF-04b) + map click (RF-06c)  
3. **Zone deals UI:** RF-10b / RF-07e vs OMI band  
4. **HPO:** RF-05b on semester-split OMI features  
5. **Stretch:** Pandera, RF-04d parallelism  

Immobiliare scrape: **removed from the codebase**; historical requirements in SOR v1.

---

## 6. Open questions (v2)

- [ ] Which municipio / OMI boundary dataset (license + stability)?  
- [ ] HPO: single model family with search, or multi-model selection?  
- [ ] Metric for promotion: MAE only, or MAE + calibration / bias by zona?  
- [ ] Public zone deals: feature rows vs aggregates-only (stricter RNF-03)?  
- [ ] Feature parallelism: threads vs processes vs vectorize-only?  
- [ ] Map library for RF-06c and reference point?  
- [x] Primary source = OMI (Immobiliare code removed)  
- [ ] OMI access: personal Fisconline vs sample semester in DVC (citation «Agenzia Entrate – OMI»)?

---

## 7. Traceability to v1

| v1 | v2 |
|----|-----|
| Immobiliare scrape ETL | OMI loader + features (`doc/omi.md`) |
| RF-04 lite listing features | RF-04b geo zona/municipio |
| RF-05 single HGB | RF-05b selection / HPO |
| RF-10 listing good deals | RF-10b zone below/above band |
| RF-06 listing predict form | OMI zone form; RF-06c map later |
| Daily scrape CI | Manual download + DVC + `omi-monitoring` workflow |
