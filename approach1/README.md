# Dhaka Next-Day PM2.5 Forecasting

> **Single entry point.** If you did not build this project, read this file top to bottom —
> it tells you what the system does, how to run it, what the results mean, and where every
> output lives. No machine-learning background assumed.
>
> 📘 **New here / want the full beginner guide?** Read **[MASTER_DOCUMENTATION.md](MASTER_DOCUMENTATION.md)** —
> a zero-to-expert walkthrough with plain-English concepts, every folder & file described, the
> full results, the annotated Streamlit-app screenshot, and why this model fits the research paper.

## Project goal (two sentences)

This project predicts **tomorrow's average PM2.5** (fine-particle air pollution, in µg/m³)
for one reference air-quality monitor in Dhaka, using only information available **today** —
recent pollution, weather, the calendar, and fire activity. The bar is **methodological
honesty under peer review**: a clean, leakage-free model that demonstrably beats naive
forecasts, not an exotic model that cannot be defended.

---

## What exactly is predicted — the single-station target

- **Target:** the **next-day (t+1) daily-mean PM2.5** at **OpenAQ sensor 24434**, the
  **US diplomatic-post reference monitor** (~2 km from the city centre), predicted from data
  through day *t*. The daily mean is taken over the **local Dhaka calendar day** (UTC+6) and
  a day counts only if it has **≥18 valid hourly readings**. This is a *temporal* forecast at
  one point — not a map, not a same-hour shift, not an AQI.
- **Why one raw sensor (important):** the merged dataset ships a column `openaq_pm25`, but
  Phase-0 verification proved it is an **hourly average over ~14 monitoring sites spread
  across ~17 km** — not a single station. Averaging distant sites would break the
  single-point framing, so we take the target straight from **one sensor (24434)** instead.
- **Cost of that honesty:** the sensor reported continuously 2016→2025 but went offline after
  **2025-03-24**, giving **N ≈ 1,746** supervised (today, tomorrow) day-pairs. That is a small
  sample; we report it plainly and temper expectations for data-hungry models accordingly.

---

## Quick start (copy-paste)

```bash
# --- 0. Install (Python 3.11+) ---
python -m pip install -r requirements.txt
# optional: freeze exact versions for a reproducibility record -> pip freeze > requirements.lock

# --- 1. Reproduce the full pipeline, in order ---
python src/build_dataset.py     # Phases 2-3: build processed/supervised_daily.parquet
python src/run_baselines.py     # Phase 4:   naive baselines (the bar to beat)
python src/run_models.py        # Phase 5:   LightGBM/XGBoost/RF + Optuna + walk-forward CV
python src/run_deep.py          # Phase 6:   Bi-LSTM+attention comparison (slow, CPU ~15 min)
python src/run_shap.py          # Phase 7:   SHAP explainability + figures
python src/run_ablations.py     # Phase 8:   feature-group ablations
python src/run_self_audit.py    # Final:     leakage/repro checklist (expect 23/23 PASS)

# --- 2. Launch the local proof-of-concept app ---
streamlit run app/streamlit_app.py
```

Every setting (random seed, ≥18-hour rule, forecast horizon, walk-forward scheme, feature
groups, model search spaces, AQI breakpoints) lives in **`config/config.yaml`** — the single
source of truth. There are no magic numbers in the code.

---

## Results in one paragraph

The primary model, **LightGBM**, achieves a **hold-out RMSE of 30.25 µg/m³** (R² 0.81) on the
untouched final 12 months, beating the **persistence** baseline ("tomorrow = today") by a
**+0.15 skill score** and **seasonal climatology** by **+0.32** — i.e. a modest but real edge
over the hard autocorrelation baseline and a large edge over the seasonal average. **SHAP**
attributes ~**58%** of the model's decisions to **meteorology** (boundary-layer height, wind,
humidity), ~37% to PM2.5 memory (lags/rolling), and essentially nothing to fire. Ablations
confirm the PM2.5 lag/rolling features are what lift the model over persistence, and that the
near-circular CAMS `omaq_pm2_5` field adds nothing (so it is excluded from the default model).
The **Bi-LSTM+attention** deep model **underperforms** LightGBM and even persistence on
cross-validation at this small N (≈1,746) — an expected, honestly-reported null result, not a
failure to hide.

| | Walk-forward CV RMSE | Hold-out RMSE | Skill vs persistence |
|---|---|---|---|
| **LightGBM (primary)** | 30.26 | **30.25** | +0.15 (hold-out) |
| XGBoost | 30.83 | 31.17 | +0.12 |
| RandomForest | 30.77 | 30.97 | +0.13 |
| persistence baseline | 33.12 | 35.58 | — |
| climatology baseline | 44.45 | 36.78 | — |
| Bi-LSTM+attention | ~36.97 | 30.73 | underperforms at this N |

---

## How leakage is prevented (why you can trust the numbers)

"Leakage" = letting the model peek at data it would not have at prediction time; it inflates
results and ruins reproducibility. Defenses, all enforced in code:

1. A feature for day t+1 may use data only **through day t** (only deterministic *calendar*
   fields of t+1, knowable in advance, look forward).
2. **Walk-forward (expanding-window) validation** — train on the past, test on the strictly
   later future. No KFold, no shuffling.
3. Imputer medians, the IsolationForest anomaly flag, and scaling are fit on each fold's
   **train rows only**; deep-model early stopping uses an inner slice of the train, never the test.
4. The most recent **12 months are split off first** and scored **once** — Optuna never sees them.
5. The target's future is never imputed; the near-circular `omaq_pm2_5` is excluded from the
   default features (used only in a clearly-labelled ablation).

---

## File map — what each output contains

### Reports (`reports/`)
| File | What it contains |
|---|---|
| `phase0_data_audit.md` | Data-verification audit; the single-station finding and the decision to use sensor 24434; sensor-24434 coverage profile. |
| `phase2_feature_dictionary.md` | Every engineered feature, its ablation group, and its temporal availability (the leakage justification). |
| `phase3_target.md` | Exact target definition; qualifying-day and supervised-pair counts. |
| `phase4_baselines.md` | Persistence / climatology / seasonal-naive RMSE/MAE/sMAPE/R² (the bar every model must beat). |
| `phase5_models.md` | LightGBM/XGBoost/RF CV + hold-out metrics, skill scores, best Optuna params, verdict. |
| `phase6_deep_comparison.md` | Bi-LSTM+attention vs trees, with the small-N caveat (the null result). |
| `phase7_results.md` | SHAP feature-group attribution and top features (where the skill comes from). |
| `phase8_ablations.md` | Feature-group ablation table + the `omaq_pm2_5` circularity caveat. |
| `self_audit.md` | Final pass/fail over the leakage/reproducibility checklist + non-negotiable constraints (23/23). |
| `phase0_*.py` | Read-only scripts that reproduce the Phase-0 audit numbers. |

### Artifacts (`artifacts/`)
| File | What it contains |
|---|---|
| `model_lightgbm.joblib` | The **primary** trained model + its fold-fitted preprocessing (medians, IsolationForest). Load with `joblib`. |
| `model_xgboost.joblib`, `model_random_forest.joblib` | The secondary trained models, same bundle format. |
| `lgbm_best_params.json` | LightGBM's Optuna-selected hyperparameters (reused by the ablations). |

### Data & figures
| Path | What it contains |
|---|---|
| `processed/supervised_daily.parquet` | The model-ready table: one row per origin day t, 62 columns (features + next-day label). Built by `src/build_dataset.py`; the raw master is never overwritten. |
| `processed/feature_groups.json`, `processed/baseline_rmse.json` | Feature→group map and baseline RMSEs used downstream. |
| `figures/shap_beeswarm.png`, `shap_bar.png`, `shap_group_attribution.png` | SHAP plots for the manuscript. |

### Code (`src/`) and app
| File | Role |
|---|---|
| `utils.py` | Config loading, global seed, paths. |
| `data_loader.py` | Loads the hourly master (features) and the single-station sensor (target). |
| `targets.py` | Daily-mean target + next-day labels. |
| `features.py` | Leakage-safe daily feature engineering. |
| `validation.py` | Expanding walk-forward split + anchored 12-month hold-out. |
| `baselines.py` / `evaluate.py` | Naive baselines / metrics + skill scores. |
| `models_tree.py` | Gradient boosting + Optuna + fold-internal preprocessing. |
| `models_deep.py` | Bi-LSTM+attention comparison model. |
| `run_*.py` | One driver per phase (see Quick start). |
| `app/streamlit_app.py` | Local proof-of-concept UI (loads the LightGBM bundle). |

### Session memory (`.project_memory/`)
`architecture_decisions.md` (dated log of decisions + rationale) and `state_tracker.json`
(machine-readable phase status) — so the work can be picked up after a break.

---

## Methods paragraph (manuscript draft)

> We forecast next-day daily-mean PM2.5 at a single Dhaka reference monitor (OpenAQ sensor
> 24434, the US diplomatic-post station, ~2 km from the city centre) over 2016–2025. The target
> is the mean of measured hourly PM2.5 on the local calendar day (Asia/Dhaka, UTC+6), retained
> only for days with ≥18 valid hours, yielding 1,746 consecutive (t, t+1) day-pairs. Predictors,
> all available through day t, comprise ERA5 reanalysis meteorology, deterministic
> calendar/cyclical features of the target day, autoregressive PM2.5 lags (1/2/3/7 days) and
> trailing rolling statistics (3/7/14-day mean/std/min/max), and NASA FIRMS active-fire counts
> and radiative power; the near-static ACAG satellite annual grid serves only as a coarse
> anchor. Models (LightGBM, XGBoost, RandomForest) were tuned with Optuna optimising an
> expanding-window walk-forward cross-validation RMSE, with all learned preprocessing fit on
> training folds only; a 12-month hold-out anchored to the last available day was evaluated
> once. Skill is reported relative to persistence and day-of-year climatology baselines. A
> compact Bi-LSTM with attention is included as an honest, sample-size-limited comparison, not
> the headline. Data provenance: OpenAQ in-situ measurements; ERA5 and NASA POWER meteorology;
> NASA FIRMS fire detections; ACAG (van Donkelaar et al.) satellite PM2.5.
