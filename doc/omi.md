# OMI Open Data

Primary market source: **Agenzia delle Entrate — Osservatorio del Mercato Immobiliare (OMI)**.  
Cite in UI/docs: «Agenzia Entrate – OMI».

Download is **manual** (Fisconline / Entratel). Cloud/CI pulls CSVs via DVC and runs the rest of the pipeline (fails if remote has no OMI CSV).

## Download (Forniture OMI)

1. Area riservata Fisconline/Entratel → **Forniture OMI – Quotazioni Immobiliari**.  
2. Richiedi territorio (comune **Roma** *oppure* provincia **RM**) + **semestre**.  
3. Nella pagina “richieste non scadute”, aspetta il **file elaborato** e scarica.  
4. Ripeti per **≥2–3 semestri** (es. 2025/2, 2025/1, 2024/2).

Official entry points:

- [Quotazioni immobiliari (cittadini)](https://www.agenziaentrate.gov.it/portale/schede/fabbricatiterreni/omi/banche-dati/quotazioni-immobiliari-cittadini)  
- [Forniture dati OMI](https://www.agenziaentrate.gov.it/portale/schede/fabbricatiterreni/omi/forniture-dati-omi-imprese) — Fisconline / Entratel  

### Provincia vs comune

Una fornitura **provincia RM** contiene tutti i comuni (Roma, Fiumicino, Tivoli, …).  
Il loader **filtra solo il comune di Roma** (`Comune_descrizione` contiene `"roma"`). Il resto provincia viene ignorato.  
Se la Fornitura permette **comune = Roma**, il CSV è già più piccolo e allineato al prodotto.

### File ufficiali

Ogni elaborazione tipicamente dà:

| File | Ruolo |
|------|--------|
| `QI_*_<YYYY><S>_VALORI.csv` | Quotazioni (locazione + compravendita) — **usato dal pipeline** |
| `QI_*_ZONE.csv` | Info zone — **ignorato** se è presente almeno un `*VALORI*` |

Esempio in repo:

```text
data/raw/omi/
  QI_1428681_1_20252_VALORI.csv   # semestre 2025-2 (da nome file o titolo)
  QI_1428681_1_20252_ZONE.csv     # opzionale
  QI_…_20251_VALORI.csv           # altri semestri accanto
```

Il file VALORI ha spesso una **riga titolo** prima dell’header; il loader la salta.  
Semestre: da titolo (`Semestre 2025/2`) o da stem (`…20252…` → `2025-2`).

`data/raw/omi/` is gitignored (sync with DVC). Unit tests use tiny inline CSV strings — not a fixtures directory.

## Expected columns (flexible names)

| Logical field | Typical names (export ufficiale) |
|---------------|----------------------------------|
| comune | `Comune_descrizione`, `COMUNE` |
| zona | `Zona`, `ZONA` |
| zona_descr | `Zona_Descr` (più spesso nel file ZONE) |
| tipologia | `Descr_Tipologia` |
| stato | `Stato` |
| loc_min / loc_max | `Loc_min`, `Loc_max` (decimali con `,`) |

Kept by default: residential / abitazioni with valid loc_min/loc_max; non-residential (negozio, ufficio, …) dropped.

## Pipeline

```bash
# After placing real VALORI CSVs in data/raw/omi/
.venv/bin/python run_pipeline.py -v
```

Stages: `etl.extract.omi_loader` → `etl.transform.omi_features` → `ml.train`.

**Target:** `price_per_m2_monthly` = mid `(loc_min + loc_max) / 2`.  
**Features:** `zona_omi`, `tipologia`, `stato`; plus `publication_month` / `loc_mid_lag` when ≥2 semesters give signal.  
Do **not** feed current `loc_min`/`loc_max` into the model (leakage).  
With a **single** semester, train still runs (numeric lag/month dropped) but split is ordered holdout, not temporal.  
With **two** semesters, temporal split puts only the earlier semester in train → lag/month often still dropped on that fold; prefer **≥3 semesters** for full numeric signal.

## Sync to cloud (short)

1. Local: CSVs in `data/raw/omi/` → `run_pipeline.py -v` (optional).  
2. `dvc add data/raw` + `dvc add data/processed/features_latest.jsonl` → `dvc push` → commit `.dvc` pointers.  
3. Commit/push **code** + `models/baseline_latest/` (API bake).  
4. Render: redeploy API (+ UI if needed).  
5. GitHub Actions → *omi-monitoring* → Run workflow (optional drift/snapshots).

Details: [`dvc.md`](dvc.md), [`render.md`](render.md), [`usage.md`](usage.md).

## Product

- Predict: fair €/m² from zone / typology / conservation (+ lag).  
- “Deals”: zones below/above model (±10%) — not listing ads.  
- **Attribution:** always «Agenzia Entrate – OMI» in UI/docs.  
- **Public UI / git snapshots:** zone labels + `deal_label` only — **no** OMI locazione €/m² (mid/min/max), no gap residuals. Raw CSVs stay private (gitignored / DVC).
