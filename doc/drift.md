# Evidently drift report

`ml/drift_report.py` (RF-08 lite): confronta **reference** (“prima”) vs **current** (“ora”).  
Comandi: [`usage.md`](usage.md). SOR: RF-08.

**Cosa fa:** drift sulle *forme* di feature / target / predizioni; con modello, anche **MAE/RMSE per giorno** di `scraped_at` (proxy di `P(y|x)` / semestre OMI).  
**Cosa non fa:** retrain automatico → [`ml.retrain_check`](../ml/retrain_check.py) (RF-09); metriche di training → `ml.train`.

**RF-09 (gate MAE):** dopo il report, `python -m ml.retrain_check` ritrena se `mae_current / mae_reference >= 1.5` e `n_reference >= 50`. Drift di feature/prediction solo come contesto in `decision.json`, mai come trigger.

---

## Teoria in breve

```text
P(x,y) = P(y|x) P(x) = P(x|y) P(y)
```

| Quantità | Significato | Come la conosci |
|----------|-------------|-----------------|
| `P(x)` | Mix feature | Test drift per colonna |
| `P(y)` | Mix €/m² | Drift su `price_per_m2_monthly` |
| `P(ŷ)` | Mix output modello | Drift su `prediction` |
| `P(y∣x)` | €/m² **dato** il flat | Modello `f(x)`; `mae_reference` / `mae_current` + `mae_by_day` |
| `P(x∣y)` | Feature **dato** un canone | Bin di €/m² + confronto profili (vedi sezione sotto); non ancora nel codice |

---

## Tipi di drift (con esempi Roma)

Casi **ideali** ([Chip Huyen](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html)); in pratica si sovrappongono ([Evidently](https://www.evidentlyai.com/ml-in-production/concept-drift)).  
La decisione **retrain** segue soprattutto l’errore (MAE qui; accuracy/AUC in classificazione), non il solo nome dello shift.

| Tipo | Cambia | Resta (per def.) | Esempio Roma | Segnale | Azione tipica |
|------|--------|------------------|--------------|---------|---------------|
| **Covariate shift** | `P(x)` | `P(y∣x)` | Più bilocali in centro; mix diverso, a parità di flat il prezzo equo è lo stesso | Feature flagged; `prediction` può muoversi *di conseguenza*; **MAE ok** | Non retrain (il modello “assorbe” il nuovo mix) |
| **Label / prior shift** | `P(y)` | `P(x∣y)` — stesso “look” dentro ogni fascia/classe | Canoni medi su, ma il profilo tipico a 20€/m² **è ancora lo stesso** | Target flagged; feature *dentro* le fasce stabili | Spesso ricalibra (prob/soglia); retrain solo se l’errore decisionale sale |
| **Concept drift** | `P(y∣x)` (oltre un puro cambio di prior) | spesso `P(x)` simile | Stesso bilocale a Trastevere, mercato più alto → sottostima sistematica | Feature stabili ma `mae_current` ≫ `mae_reference` | **Riallena** |
| *(non è una riga “da manuale”)* **Mix dentro le label** | `P(x∣y)` | — | Stessa fascia di €/m², ma profili `x` diversi (nuove zone/tipologie nello stesso canone) | Feature diverse *condizionando* a `y`; **MAE ok** | Non è prior shift (lì `P(x∣y)` resta fisso). Se l’errore resta basso → **non** retrain: `x → y` tiene comunque |

Nota: “stesse bande di `y`, profili diversi” **rompe** l’invariante del prior shift; non contraddice covariate/concept — è un altro fatto empirico. Prior shift = frequenze di `y` diverse, **faccia** delle classi uguale.

Oggi misuriamo **marginali univariate** ([DataDriftPreset](https://docs.evidentlyai.com/metrics/preset_data_drift)) + MAE per giorno:

```text
P(x_i), P(y), P(ŷ)  ✓     |     P(y|x) ≈ mae_ref/cur + mae_by_day  ✓     |     P(x|y)  ✗ (bin non ancora)
```

`prediction` drift = **proxy** (“investiga”), non prova di concept drift.  
Con modello: `mae_reference` / `mae_current` sullo stesso split del drift; `mae_by_day` = errore per `scraped_at` day. Con `--no-model` sono `null`.

---

## Profili multivariati: come controllare `P(x|y)`

Parlare di “stesso profilo dentro una fascia di `y`” significa confrontare la **congiunta** `P(x | y = y_k)` (o, in regressione, `P(x | y ∈ bin_k)`), non una sola feature.

### Perché le marginali non bastano

Se `x = (x_1, …, x_d)`, un test univariato guarda `P(x_j)` o `P(x_j|y)` per ogni `j`.

- Marginali uguali **non** implicano congiunte uguali: possono cambiare le **dipendenze** (es. `zona_omi` e `loc_mid_lag` vs mid target).
- Una marginale in drift **non** dice da sola se è covariate shift, prior shift o mix dentro le label: serve il *condizionamento* a `y` e/o la performance.

`DataDriftPreset` (come oggi nel repo) = tipicamente **marginali per colonna** sul dataset intero — utile e standard, ma **non** è un test del joint `P(x|y)`.

### Come si confrontano i profili in pratica

Fissa fasce/bin di `y` (classi, o bin di €/m²). **Dentro ogni fascia**, confronta gli `x` di reference vs current:

| Metodo | Cosa stima | Note |
|--------|------------|------|
| Marginali **per feature, dentro il bin** | pezzi di `P(x_j∣y)` | Minimo utile per “prior shift vs mix dentro le label”; ancora non è il joint |
| Correlazioni / coppie dentro il bin | un pezzo di dipendenza | Complemento alle marginali; incompleto in alto `d` |
| Distanza multivariata (MMD, energy distance, …) sugli `x` del bin | `P(x∣y)` come cloud | Two-sample sul vettore `x`; serve abbastanza punti per bin ([Gretton et al., MMD](https://jmlr.org/papers/v13/gretton12a.html)) |
| **Classifier two-sample** | stessa idea, spesso più pratica in alto `d` | Alleni un classificatore *ref vs cur* solo su `x` **dentro** il bin. AUC ≈ 0.5 → profili indistinguibili; AUC alto → `P(x∣y)` è cambiata |

Procedura classifier two-sample (idea): label fittizia `source ∈ {ref, cur}` → fit su `x` → AUC (o accuracy) fuori sample. Non stima densità; chiede solo se i due campioni sono separabili.

### Cosa *non* confondere

| Domanda | Quantità | Strumento adatto |
|---------|----------|------------------|
| Il “look” dentro le classi/fasce è lo stesso? | `P(x∣y)` | Bin di `y` + MMD / classifier two-sample (o marginali in-bin) |
| Il meccanismo predittivo tiene? | utilità di `P(y∣x)` / `ŷ` | **Errore** sul current (MAE/AUC) — già un riassunto *multivariato* di `x → y` |
| `P(y∣x)` ref vs cur in senso stretto | uguaglianza delle condizionali | Test condizionali (es. varianti di conditional MMD); più pesanti; in produzione si usa spesso l’errore come proxy |

Quindi: **controllare i profili** ≠ guardare una sola marginale globale.  
In multi-D: o confronti il joint (dentro i bin di `y`), o — per la decisione retrain — lasci che l’**errore** ti dica se il profilo che conta per predire è ancora valido.

### In questo repo

- Oggi: marginali univariate globali + MAE (proxy su `P(y|x)`).  
- Non ancora: bin di €/m² + confronto `P(x|y)` (né univariato in-bin né classifier two-sample).

---

## Calibrazione dei test Evidently

I metodi di default **cambiano con la size della reference** ([docs](https://docs.evidentlyai.com/metrics/customize_data_drift)):

| Reference | Numeriche | Categoriche | Soglia tipica |
|-----------|-----------|-------------|---------------|
| ≤ 1000 righe (oggi) | K-S / Z-test | Chi-square | p &lt; 0.05 |
| &gt; 1000 righe | Wasserstein | Jensen–Shannon | distanza ≈ 0.1 |

Uno storico di `drifted_columns_share` **non è confrontabile 1:1** prima/dopo i 1000: il report “si ricalibra” da solo. Meglio fissare metodi/soglie espliciti quando i dati crescono.

---

## Prior shift vs concept drift (Bayes)

Da Bayes:

```text
P(y|x) = P(x|y) P(y) / P(x)
```

Se cambia solo la **prevalenza** `P(y)` e resta fisso `P(x|y)`, allora `P(y|x)` **deve** cambiare.  
Quindi: **prior / label shift implica uno spostamento del posteriore**. Non è una contraddizione.

Perché allora non lo chiamiamo sempre concept drift? La differenza è sul **meccanismo**, non sul fatto che `P(y|x)` si muova.

| | **Prior / label shift** | **Concept drift** (senso operativo) |
|--|-------------------------|-------------------------------------|
| Invariante | `P(x∣y)` — il “look” di ogni classe (o fascia di `y`) | spesso (idealmente) `P(x)` |
| Cosa cambia alla radice | solo quanto è frequente `y` | *come* `x` si collega a `y` (significato / meccanismo) |
| `P(y∣x)` | cambia **solo perché** è cambiato `P(y)` (Bayes) | cambia **oltre** un semplice cambio di prior |
| Azione tipica | ricalibrare probabilità / soglia su un campione del regime attuale | riallenare il modello |

Esempio classificazione binaria: prevalenza frodi 5% → 20%, ma una frode “sembra” ancora uguale → prior shift → ranking (AUC) ok, probabilità miscalibrate → **ricalibra**.  
Stesse feature, etichetta “giusta” diversa (lo spam di ieri è legittimo oggi) → concept drift → AUC giù → **riallena**.

Le tassonomie non sono ortogonali al 100%: prior shift *induce* un cambio di `P(y|x)`. “Concept drift” di solito riserva il nome al caso in cui **non** basta aggiornare prior/calibrazione.

In questo repo (regressione): segnale forte per retrain ≈ `mae_current` ≫ `mae_reference` con feature stabili; solo drift di mix/forme con MAE stabile → spesso non retrain (investiga / aggiorna reference). Gate automatico: `python -m ml.retrain_check` (soglia default 1.5×, min reference 50).

**Letture:** [data drift](https://www.evidentlyai.com/ml-in-production/data-drift) · [concept drift](https://www.evidentlyai.com/ml-in-production/concept-drift) · [Chip Huyen](https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html) · [MMD two-sample (Gretton et al.)](https://jmlr.org/papers/v13/gretton12a.html)
