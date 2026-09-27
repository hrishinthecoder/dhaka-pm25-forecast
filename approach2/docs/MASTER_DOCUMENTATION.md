# Dhaka Next-Day PM2.5 Forecasting — A Complete Beginner's Guide

> **Read this first.** This guide explains the whole project from zero. You do
> **not** need to know machine learning, statistics, or air-quality science to
> follow it. Technical terms are written in **plain words first**, then the exact
> numbers and methods are given so an expert can check them too.
>
> Every number, table, and caveat in the "Results" and "Model card" sections is
> reported exactly as the pipeline produced it — nothing is rounded, softened, or
> exaggerated for this guide.

---

## Table of contents

- [Part 0 — The 30-second summary](#part-0--the-30-second-summary)
- [Part 1 — The 5 W's + How (start here)](#part-1--the-5-ws--how-start-here)
- [Part 2 — Plain-language glossary](#part-2--plain-language-glossary)
- [Part 3 — How the system works, step by step](#part-3--how-the-system-works-step-by-step)
- [Part 4 — Where the data comes from (provenance)](#part-4--where-the-data-comes-from-provenance)
- [Part 5 — What the model looks at, and the rules it must obey](#part-5--what-the-model-looks-at-and-the-rules-it-must-obey)
- [Part 6 — How we trained and tuned it](#part-6--how-we-trained-and-tuned-it)
- [Part 7 — Results, and how to read them honestly](#part-7--results-and-how-to-read-them-honestly)
- [Part 8 — Limitations & honest claims](#part-8--limitations--honest-claims)
- [Part 9 — The frozen model (the official "card")](#part-9--the-frozen-model-the-official-card)
- [Part 10 — The demo app (Streamlit), page by page](#part-10--the-demo-app-streamlit-page-by-page)
- [Part 11 — How to run it yourself (reproducibility)](#part-11--how-to-run-it-yourself-reproducibility)

---

## Part 0 — The 30-second summary

This project builds a computer model that, **given today's air and weather, guesses
tomorrow's average air-pollution level in Dhaka, Bangladesh.** Pollution is measured
as **PM2.5** (tiny particles in the air that are bad for your lungs). The guess is a
number in µg/m³ (micrograms per cubic metre) plus a colour-coded health category.

It is a **research demonstration**, not a public weather service. It works on **one
measuring station in Dhaka**, looks **one day ahead**, and is honest about how often
it actually beats a dumb "tomorrow = today" guess (answer: **clearly only in the two
most recent years tested**).

---

## Part 1 — The 5 W's + How (start here)

### WHAT is this?
A **next-day PM2.5 forecaster**. You feed it information available by the end of
**today** (call it *day t*); it outputs **tomorrow's** (*day t+1*) **24-hour average
PM2.5** in µg/m³, plus the matching **US-EPA air-quality category** (Good, Moderate,
Unhealthy, etc.). The "brain" is a machine-learning model called **XGBoost**.

### WHY does it exist?
Dhaka has some of the worst air pollution in the world, and it spikes in the dry
winter season. A reliable next-day estimate could help researchers and agencies
**anticipate bad-air days**. This project is a **proof of concept**: can simple,
publicly available data predict tomorrow's pollution better than just assuming
"tomorrow will be like today"? The honest answer is "somewhat, and mostly in recent
years" — and the whole project is built to show that *honestly*, with statistics, not
hype.

### WHO is it for?
- **Researchers and collaborators** (e.g. Department of Environment, BAPA, BUET) who
  want a transparent, checkable baseline for next-day PM2.5.
- It is **not** for the general public as a live advisory, and **not** a replacement
  for official government air-quality monitoring (see [Part 8](#part-8--limitations--honest-claims)).

### WHEN (what time span)?
- **Data covers 2016 → 2025**, hour by hour.
- The model **learns** from 2016–2023, is **tuned** using up to 2024, and is **tested**
  on 2025 (data it never saw during training). It predicts **one day ahead**.

### WHERE?
- **One reference-grade monitoring station in Dhaka** (location ≈ 23.81°N, 90.41°E).
  Weather comes from a global reanalysis dataset (ERA5) for that location. Because it
  is a single point, the model does **not** know about pollution differences across
  different neighbourhoods of Dhaka.

### HOW (in one paragraph)?
We collect hourly pollution + weather + fire data, turn it into **one row per day** of
"summary numbers" (averages, recent trends, yesterday's values, etc.), and train a
model to map **today's row → tomorrow's pollution**. We then test it on a year it never
saw, and use proper statistics to ask "is it *really* better than guessing, or did it
just get lucky?" Finally we **freeze** the exact winning model so anyone can reproduce
the same predictions. The rest of this guide unpacks each step.

---

## Part 2 — Plain-language glossary

| Term | What it means in plain words |
|---|---|
| **PM2.5** | Particulate matter smaller than 2.5 microns — soot/dust tiny enough to enter your lungs and blood. Measured in µg/m³. Higher = worse air. |
| **µg/m³** | Micrograms per cubic metre — the unit for how much PM2.5 is in the air. |
| **AQI category** | The US-EPA colour bands (Good → Hazardous) that translate a PM2.5 number into a health message. |
| **Day *t* / day *t+1*** | "Today" and "tomorrow." The model uses day *t* to predict day *t+1*. |
| **Feature** | One input number the model looks at (e.g. today's average PM2.5, today's temperature). This model uses **45** of them. |
| **Target** | The thing being predicted: tomorrow's 24-hour average PM2.5. |
| **Model / XGBoost** | The "brain." XGBoost is a popular algorithm that combines many small decision trees. |
| **Baseline** | A dumb reference to beat. **Persistence** = "tomorrow equals today." **Climatology** = "tomorrow equals the historical average for this day of year." A good model must beat these. |
| **Training / tuning / test** | *Training*: the model learns patterns. *Tuning*: we pick the model's settings. *Test*: we check it on unseen data (2025) to get an honest score. |
| **Leakage** | Accidentally letting the model peek at information it wouldn't really have at forecast time. Leakage makes scores look great but is cheating; we guard against it heavily. |
| **RMSE / MAE** | Error measures (lower = better), both in µg/m³. RMSE punishes big misses more; MAE is the plain average miss. |
| **R²** | "How much of the up-and-down pattern the model explains," from 0 (useless) to 1 (perfect). |
| **Bootstrap confidence interval (CI)** | A range like `[28.2, 38.7]` showing the uncertainty in a score. Computed by re-sampling the test data many times. |
| **Paired test / Diebold–Mariano (DM)** | Statistics that compare the model against persistence **day by day on the same days**, to judge whether the difference is real or chance. A **p-value** near/below 0.05 means "probably real." |
| **ERA5 / Open-Meteo / CDS** | ERA5 is a global "reanalysis" weather record. We get it via Open-Meteo (easy API) and, for one gap, directly from the official Copernicus CDS archive. |
| **BLH (boundary-layer height)** | How tall the layer of air near the ground is. A low BLH traps pollution; it matters a lot for PM2.5. |
| **FIRMS** | NASA satellite data on active fires (smoke is a pollution source). |
| **Reanalysis vs forecast** | Reanalysis = the *best after-the-fact* reconstruction of weather. A real operational system would instead need a *forecast* (NWP) of tomorrow's weather. This project uses reanalysis as a stand-in. |
| **Frozen model** | The exact, locked final model saved with checksums so predictions can be reproduced byte-for-byte. |

---

## Part 3 — How the system works, step by step

Think of it as an assembly line. Each stage has a folder/script.

1. **Collect the raw data (ingest).** Pull hourly PM2.5 (OpenAQ), weather (ERA5 via
   Open-Meteo), fires (NASA FIRMS), and calendar facts (weekends, holidays, Ramadan).
   Everything is cached so re-runs are identical. → `src/ingest/`, `notebook 01`.
2. **Fix one weather gap.** One weather variable (BLH) was missing for Jan–Jun 2024 in
   the easy data source, so we pull those exact hours from the official archive (CDS).
   This happens automatically during ingest. → `src/ingest/cds_blh.py`, `patch_blh.py`.
3. **Build daily features.** Turn messy hourly data into **one tidy row per day** of 45
   summary numbers (today's average, recent trends, weather averages, fire counts,
   calendar flags). Carefully avoid leakage. → `src/features/daily.py`.
4. **Train models.** Teach several models (Random Forest, XGBoost, LightGBM) plus the
   two dumb baselines. → `src/models/train.py`.
5. **Evaluate honestly.** Score everything on the unseen 2025 data, with uncertainty
   ranges and paired statistical tests; also run a 6-year "rolling backtest." →
   `src/evaluate/`.
6. **Freeze the winner.** Lock the chosen XGBoost model + checksums. →
   `src/freeze_model.py`, `models/final/`.
7. **Predict / demo.** Ask for any date and get tomorrow's forecast; or open the
   Streamlit app to explore it. → `src/predict.py`, `app/`.

You can read the exact commands for every stage in
[Part 11](#part-11--how-to-run-it-yourself-reproducibility), and explore the model in
the demo app described in [Part 10](#part-10--the-demo-app-streamlit-page-by-page).

---

## Part 4 — Where the data comes from (provenance)

The model predicts the **next-day (t+1) 24-hour mean PM2.5** (µg/m³) at a single Dhaka
reference station — **OpenAQ sensor 24434 (US Diplomatic Post, ~2 km from centre)** —
from information available through the end of day *t*. Secondary
output: next-day **US-EPA 2024 24-h PM2.5 AQI category**.

| Source | Variables | Provenance / notes |
|---|---|---|
| **OpenAQ v3** | ground-truth PM2.5 (target), o3, pm10 | Reference-grade monitors only; PM2.5 is the target, o3/pm10 are **excluded** as predictors (same station, not forecast-available). Single-station scope. |
| **ERA5 via Open-Meteo** | 2 m temp, RH, dew point, surface pressure, cloud cover, wind speed+direction, precip, shortwave radiation, **BLH** | Reanalysis, used as a day-t predictor proxy. |
| **ERA5 via Copernicus CDS** | **boundary-layer height, 2024-01-01→06-30** | The Open-Meteo *mirror* had a gap for this window; the variable was pulled **directly from the native CDS archive** (`src/ingest/cds_blh.py`), patched in by `src/ingest/patch_blh.py`. Real-reanalysis window mean 492.85 m vs the prior climatological imputation 496.9 m (<1% — the imputation introduced no material distortion). 0 imputed BLH hours remain. |
| **NASA FIRMS** | fire count, FRP sum | Daily fire activity, broadcast to hours then re-aggregated daily; lags {0,1,2}. |
| **Calendar** | day_of_week, month, is_weekend, is_holiday, is_ramadan | Deterministic, known ahead. |

Single station (sensor 24434). The master keeps a continuous hourly UTC index
2016-01-01 → 2025-12-31 (87,672 rows); the PM2.5 **target** is valid 2016-11 → 2025-03-24
(sensor 24434's own record), NaN elsewhere — never filled from other stations.

**Beginner note:** "single station" and "no traffic data" are real limits — keep them
in mind when reading the results.

---

## Part 5 — What the model looks at, and the rules it must obey

### What it predicts (the target), in plain terms
- We average all the hourly PM2.5 readings within a day to get one daily number.
- A day only "counts" if it has **at least 18 of 24 hours** of readings (so we don't
  trust a day with barely any data).
- We only keep a (today → tomorrow) pair if **both** days count.
- If any input is missing, we **drop that day rather than guess** — the model never
  invents history.

*Exact rules:*
- Daily mean PM2.5 is **valid** only with **≥18 hourly observations** that day
  (`coverage.min_hourly_obs_per_day`).
- A (t → t+1) pair is kept only if **both** days are valid (`require_both_days`).
- Rows missing any feature (e.g. a lag/rolling value near a coverage gap) are
  **dropped, never imputed** — no fabricated predictor history.

### The 45 features (the inputs), grouped
Full list: `reports/feature_availability.csv`.

| Group | Count | Availability | Members |
|---|---|---|---|
| PM2.5 history | 9 | day t and earlier | `pm25_t`, lags {1,2,3,7}, rolling mean+std {3,7} |
| ERA5 met (mean/min/max + sums) | 23 | day t | temp, RH, dew point, surface pressure, cloud cover, wind speed, BLH (×3 aggs); precip + shortwave (daily sum) |
| ERA5 wind vector | 2 | day t | `wind_u_mean`, `wind_v_mean` (decomposed from speed+direction; handles circularity) |
| FIRMS fire | 6 | day t / t-1 / t-2 | `firms_count`, `firms_frp_sum` at lags {0,1,2} |
| Calendar | 5 | known ahead | day_of_week, month, is_weekend, is_holiday, is_ramadan |

**Dead feature removed:** `era5_blh_imputed_day` was constant 0 after the CDS patch
(no imputed hours remain) and was dropped — 46 → **45**. The hourly `era5_blh_imputed`
provenance flag is retained in the parquet but not fed to the model.

### The anti-cheating rules (leakage controls)
In plain words: the model is only allowed to see things it would genuinely know by the
end of today. We block several tempting shortcuts:
- **No t+1 fields** anywhere in the feature matrix.
- `openaq_o3`, `openaq_pm10` **excluded** — same monitoring station as the target,
  not guaranteed available at forecast time (enforced in `src/features/daily.py`,
  raises if they leak in).
- **No same-period satellite PM2.5** (`omaq_*`, `acag_*`) as a predictor.
- Scalers fit on **train only**.

---

## Part 6 — How we trained and tuned it

### The split (and an honest caveat)
We never shuffle time. We cut history into three blocks:

| split | rule | model rows | plain meaning |
|---|---|---|---|
| pre-holdout (train + CV) | day-t < 2024-03-23 | 1,341 | what the model learns from + tunes on |
| holdout (test) | last 12 months, day-t 2024-03-23 → 2025-03-23 | 219 | the untouched exam |

**Anchored holdout:** sensor 24434 ends 2025-03-24, so a fixed "≥2025" calendar split
would leave too few test days. The holdout is instead the **last 12 months of available
labeled pairs**. Hyperparameters were selected via rolling-origin
`TimeSeriesSplit(n_splits=5)` over the **whole pre-holdout window**, and the final model
was **refit on that same window**. The **12-month holdout was untouched by any selection
step** — it is the clean hold-out. (No separate validation year now: CV and the final
refit both use the full pre-holdout window.)

> **Beginner translation:** the last 12 months (ending 2025-03) are the trustworthy
> "final exam" — the model never saw them while learning or tuning. Everything earlier
> was used both to learn and to pick settings.

### Tuning protocol
XGBoost and LightGBM tuned by `GridSearchCV` over `TimeSeriesSplit(5)`
(`neg_root_mean_squared_error`), then refit on train. Random Forest fixed
(n_estimators=500). Single seed 42. Frozen XGBoost hyperparameters:
`max_depth=3, learning_rate=0.03, n_estimators=300` (CV RMSE 31.14). The primary model is
the **lowest CV-RMSE among the tuned models** (XGBoost 31.14 < LightGBM 32.19); Random
Forest is a fixed, untuned baseline.

> **Beginner translation:** we tried a small grid of settings using only past data to
> judge each, picked the best, and used the same fixed random seed (42) so results are
> repeatable.

---

## Part 7 — Results, and how to read them honestly

### 7.1 Headline (held-out test = last 12 months, n=219; 95% bootstrap CI, n_boot=1000)
*How to read:* lower RMSE/MAE = better; higher R² = better. The `[...]` is the
uncertainty range. The two baselines (persistence, climatology) are what we must beat.

| model | RMSE | R² | MAE |
|---|---|---|---|
| **XGBoost (frozen, primary)** | 30.80 [25.93, 36.14] | 0.795 [0.73, 0.86] | 21.42 |
| Random Forest | 30.64 [25.49, 36.17] | 0.797 | 21.19 |
| LightGBM | 31.52 [26.43, 37.07] | 0.785 | 21.79 |
| persistence | 36.02 [30.38, 41.62] | 0.72 | 23.34 |
| climatology | 37.95 [32.21, 43.95] | 0.69 | 24.79 |

Marginal CIs overlap (persistence sits inside the ML RMSE CIs) — they cannot
establish the comparison; the **paired** tests below can.

> **Beginner translation:** XGBoost's error (30.8) looks lower than persistence (36.0),
> but their uncertainty ranges overlap, so the plain table alone can't prove it's
> better. That's why we use the paired test next. (Random Forest's point error is a hair
> lower, but it's the untuned baseline — we pick the primary by cross-validation, which
> chooses XGBoost.)

### 7.2 Paired significance vs persistence (bootstrap ΔRMSE/ΔMAE + Diebold–Mariano, h=1, HLN)
Δ = model − persistence; negative ⇒ model better; CI excluding 0 ⇒ beats persistence.

| model | ΔRMSE [95% CI] | ΔMAE [95% CI] | DM p |
|---|---|---|---|
| XGBoost | −5.22 [−9.49, −0.77] ✓ | −1.93 [−4.49, 0.50] | 0.022 |
| Random Forest | −5.38 [−10.18, −0.55] ✓ | −2.15 [−4.67, 0.31] | 0.034 |
| LightGBM | −4.50 [−9.19, −0.06] | −1.55 [−4.32, 0.94] | 0.063 |
| climatology (sanity) | +1.92 [−3.91, 7.94] | +1.45 [−2.03, 5.33] | 0.553 |

**Honest claim:** XGBoost and RF separate from persistence on paired ΔRMSE (CI excludes 0);
**LightGBM does not robustly separate** (its ΔRMSE CI only marginally excludes 0 and DM
p = 0.063 > 0.05). XGBoost DM p = 0.022. No model separates on ΔMAE. The gain over
persistence is real but modest, concentrated in large (episode-day) errors.

> **Beginner translation:** comparing day-by-day, XGBoost beats "tomorrow = today" on
> the RMSE measure (its advantage range does not touch zero), and the p-value (0.022) is
> below the usual 0.05 threshold. On the simpler MAE measure, no model is clearly better.
> So: a real but small edge, biggest on heavy-pollution days.

### 7.3 Rolling-origin backtest (6 folds, `reports/rolling_backtest.csv`)
We repeat the experiment six times, each time training on everything up to year *y* and
testing on year *y+1*. This answers "is the 2025 win a fluke?"

XGBoost beats persistence on paired ΔRMSE (CI excludes 0) in **3 of 6 folds (2022, 2024, 2025)**.

| test year | ΔRMSE [95% CI] | DM p | beats? |
|---|---|---|---|
| 2020 | +1.74 [−2.50, 6.09] | 0.413 | no (marginally worse) |
| 2021 | −1.81 [−9.10, 5.73] | 0.636 | no |
| 2022 | −5.47 [−10.38, −0.97] | 0.048 | **yes** |
| 2023 | +3.69 [−0.85, 8.38] | 0.139 | no (worse) |
| 2024 | −3.67 [−6.24, −1.23] | 0.012 | **yes** |
| 2025 (n=75) | −8.83 [−17.11, −1.43] | 0.035 | **yes** |

Skill over persistence is **not uniform across the backtest folds, and it is not a recency
or training-length trend** — XGBoost wins in 2022 but loses in 2023. The wins concentrate on
high-error / episode days, which is why ΔRMSE separates from persistence while ΔMAE does not.
The final fold (test 2025) has only n=75 because sensor 24434 ends 2025-03-24 — a thin,
high-variance fold. See `reports/figures/rolling_backtest_rmse.png`.

> **Beginner translation:** the model clearly beats the dumb baseline in 3 of the 6 years
> tested (2022, 2024, 2025); in 2020 and 2023 it was slightly worse. This is not about having
> more training years — 2022 wins yet 2023 loses; the edge shows up on the hardest, high-
> pollution days. We report this instead of hiding it.

### 7.4 Seasonal (XGBoost) & AQI classification
- monsoon RMSE 11.4 (R² 0.21), transition 30.0 (0.19), **winter-dry 48.0 (R² 0.32)**.
- AQI: accuracy 0.658, macro-F1 0.503.

> **Beginner translation:** the model is most accurate in the wet monsoon (cleaner,
> steadier air) and **struggles most in the dirty winter dry season** — exactly when
> accurate warnings would matter most. It puts the day in the correct AQI colour band
> about **66%** of the time.

---

## Part 8 — Limitations & honest claims

- **Single station** (sensor 24434) — does not represent spatial variability across Dhaka.
- **Small sample** — 1,560 labeled pairs; the record ends 2025-03 when the sensor went offline.
- **Thin holdout (n=219, last 12 months)** — mitigated, not eliminated, by the rolling
  backtest, which shows the persistence gap is significant in only 3 of 6 years.
- **ERA5 reanalysis-as-predictor** — operational next-day use requires substituting a
  numerical-weather-prediction (NWP) forecast for day t+1; reanalysis is not available
  in real time.
- **Winter-dry R² 0.29** — framed as a finding: skill concentrates in episode detection
  (ΔRMSE significant) rather than point accuracy (ΔMAE not significant); winter high-
  pollution variance is the hardest regime.
- LightGBM does not separate from persistence; RF and XGBoost do (paired ΔRMSE).

> **Beginner translation:** the biggest honesty points — it's one location, tested on
> one short year, uses "after-the-fact" weather (so it isn't ready for real-time use),
> and is weakest in the season that matters most. Treat it as a promising research
> baseline, not a finished product.

---

## Part 9 — The frozen model (the official "card")

This is the one-page summary of the exact, locked model saved in `models/final/`.

### Overview
- **Model:** XGBoost regressor (`max_depth=3, learning_rate=0.03, n_estimators=300`,
  `tree_method=hist`, seed 42), refit on the pre-holdout window only (day-t < 2024-03-23,
  1,341 pairs).
- **Input:** 45 day-*t* features (PM2.5 history, ERA5 met, ERA5 wind vector, FIRMS
  fire lags, calendar). See `models/final/feature_list.json`.
- **Output:** next-day (t+1) 24-h mean PM2.5 (µg/m³) + US-EPA 2024 AQI category.
- **Artifact integrity:** `models/final/SHA256SUMS` (model, scaler, feature list,
  hyperparameters, and the source parquet).

### Intended use
- **Retrospective, advisory, research** next-day PM2.5 estimation for **one Dhaka
  reference station**. Supports analysis of next-day pollution episodes and method
  benchmarking against persistence/climatology baselines.

### Out-of-scope use
- **Other cities / other stations** — single-station model, not spatially transferable.
- **Operational/real-time deployment** without substituting an **NWP forecast** for the
  ERA5 reanalysis day-t predictors (reanalysis is not available in real time).
- **Hourly or sub-daily forecasting** — the target is a daily mean (Option A).
- **Regulatory or health-critical decisions** as a standalone source.

### Performance (held-out test = last 12 months, n=219; 95% bootstrap CI)
| metric | value |
|---|---|
| RMSE | 30.80 [25.93, 36.14] µg/m³ |
| MAE | 21.42 µg/m³ |
| R² | 0.795 [0.73, 0.86] |
| vs persistence (paired ΔRMSE) | −5.22 [−9.49, −0.77] ✓ (DM p = 0.022) |
| AQI category accuracy / macro-F1 | 0.658 / 0.503 |

**Generalization caveat:** in a 6-fold rolling backtest, XGBoost beats persistence on
paired ΔRMSE in **3 of 6 years (2022, 2024, 2025)** — the next-day skill is real but not
uniform across the folds, and not a recency or training-length trend (2022 wins, 2023 loses).
Winter-dry is the weakest regime (R² 0.32); skill concentrates in episode detection (high-error
days), not point accuracy — which is why ΔRMSE separates from persistence while ΔMAE does not.

### Ethical note
This is an **advisory research tool, not a substitute for regulatory air-quality
monitoring** or official health guidance. Predictions carry substantial uncertainty
(see CIs) and degrade in high-pollution winter episodes. Do not use as the sole basis
for individual health or policy decisions.

### Provenance
Trained on the single-station CDS-patched master
`data/processed/dhaka_aq_master_singlestation.parquet` (PM2.5 target = sensor 24434; BLH
for 2024 H1 sourced from native ERA5 via Copernicus CDS). Reproduce end-to-end
per [Part 11](#part-11--how-to-run-it-yourself-reproducibility); verify artifacts via
`models/final/SHA256SUMS`.

---

## Part 10 — The demo app (Streamlit), page by page

There is a point-and-click web app for exploring the model — no coding once it is
running. Launch it from the project root:

```bash
streamlit run app/streamlit_app.py
```

It is a **display layer only**: read-only over the frozen model and the processed
data. It never retrains, never changes any file, and uses the **same** prediction code
as `python -m src.predict` (so a date in the app gives the exact same number as the
command line).

**Always-on safety features (on every page):**
- A permanent yellow banner stating this is a **retrospective research demonstration,
  not an operational advisory** (single station; uses reanalysis, not a live forecast;
  not a substitute for regulatory monitoring).
- A **checksum gate at startup**: the app re-checks every frozen file against
  `models/final/SHA256SUMS`. If anything has been altered, it shows a **red banner and
  refuses to predict** — so you can trust that what you see came from the locked model.

The left sidebar switches between four pages:

1. **Forecast** — the main interaction. Pick a day (only dates with complete inputs are
   allowed; invalid dates show a clear "no forecast" message instead of an error). You
   get **tomorrow's predicted PM2.5** (one decimal) and its **AQI band** with the
   standard EPA colour and a one-line health statement. If the real next-day value
   exists in the record, it is shown next to the prediction with the **absolute error**,
   so you see hits and misses honestly. A context line reminds you of the typical
   test-period error (and that winter errors are larger).

2. **Model performance** — the evidence, read live from the report files (never
   hard-coded): the **headline metrics table with confidence intervals**, the **paired
   significance vs persistence** table (with the plain caption that XGBoost and Random
   Forest separate from a same-as-yesterday baseline, mostly on large/episode days), the
   **seasonal skill** chart (winter-dry is hardest), and the **rolling-backtest figure**
   (skill is statistically clear only in the two most recent folds, 2024 & 2025).

3. **Recent forecasts** — a line chart of the **last 90 available days**: observed
   daily-mean PM2.5 with the model's next-day predictions overlaid, and the EPA band
   boundaries drawn as reference lines. A quick visual of where it tracks and where it
   drifts.

4. **About / model card** — renders this documentation (overview, model card, methods,
   results, reproducibility) inside the app, plus a one-line data-provenance summary.

See `app/README.md` for the launch notes and the explicit statement that **public
deployment is out of scope** for this phase.

---

## Part 11 — How to run it yourself (reproducibility)

You can rebuild **every** number above from a clean copy of the project. Single seed
**42** (`config/model.yaml`); temporal splits only, no shuffle; scalers fit on train.
Commands shown for `bash`; PowerShell equivalents noted where they differ.

> **New to terminals?** Each block below is a command you paste into a terminal opened
> in the project folder. Run them in order. Steps that pull data or train can take
> minutes.

### 11.0 Environment (install everything once)

```bash
git clone <repo> && cd "Approach two"
python -m venv .venv
source .venv/bin/activate            # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Python 3.12+ (developed on 3.14). Key libraries: pandas, numpy, scikit-learn,
xgboost, lightgbm, scipy, shap, matplotlib; `cdsapi`/`xarray`/`netCDF4` for the
native ERA5 BLH pull; `openaq` + `requests` for ingest.

### 11.1 API keys (free accounts you register for)

| Source | Where | How to get |
|---|---|---|
| OpenAQ v3 | `.env` → `OPENAQ_API_KEY` | register at https://docs.openaq.org |
| NASA FIRMS | `.env` → `FIRMS_MAP_KEY` | https://firms.modaps.eosdis.nasa.gov/api/area/ |
| Copernicus CDS | `~/.cdsapirc` (`url:` + `key:`) | register at https://cds.climate.copernicus.eu, **accept the ERA5 licence**, create the rc file |
| Open-Meteo ERA5 | none | keyless archive API |

Put your keys in a `.env` file at the project root (it is gitignored). Never commit secrets.

### 11.2 Ingest → build the master  (BLH patch is automatic)

Notebook 01 orchestrates the ingest modules (`src/ingest/{openaq,open_meteo,firms,
calendar_features,merge}.py`) and writes
`data/processed/dhaka_aq_master_rebuilt.parquet` (87,672 hourly rows × 27 cols).

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 notebooks/01_data_acquisition.ipynb
```

The merge step (`src/ingest/merge.py`) now **auto-patches the H1-2024 BLH gap** as
part of the build — no separate command needed:

- **CDS credentials present** (`~/.cdsapirc` or `CDSAPI_KEY`): pulls native ERA5
  BLH for 2024-01-01→06-30 from the Copernicus CDS (reusing the cached
  `data/interim/era5_blh_cds.nc` if present, so no re-request) and sets
  `era5_blh_imputed = 0` for those hours. **This path reproduces the published
  parquet and its `models/final/SHA256SUMS` checksum exactly.**
- **No credentials**: falls back to a train-years-only (≤2023) month×hour BLH
  climatology with `era5_blh_imputed = 1`, and prints a **WARNING** that the
  resulting parquet will **NOT** match `models/final/SHA256SUMS` (its 2024 features
  differ from the published model). Set up CDS credentials and re-run for an exact
  match.

> **Exact checksum reproduction REQUIRES CDS credentials.** Without them the
> pipeline still runs end-to-end and produces a valid (climatology-imputed) parquet,
> but it is a different artifact from the frozen, published one.

### 11.3 (Optional) Re-patch the BLH gap standalone

The patch already runs inside step 11.2. Run it directly only to re-patch an existing
parquet (same auto-selected PATH A / PATH B logic, same warning):

```bash
python -m src.ingest.patch_blh --config config/model.yaml
```

### 11.4 Main experiment (headline results + all reports)

```bash
python -m src.run_experiment --config config/model.yaml
```
Writes `reports/metrics.csv`, `reports/significance_vs_persistence.csv`,
`reports/seasonal_metrics.csv`, `reports/{shap,permutation}_importance.csv`,
`reports/classification_metrics.json`, `reports/figures/`, `reports/model_card_*.md`,
and `models/{random_forest,xgboost,lightgbm}.joblib` + `scaler.joblib`.

### 11.5 Rolling-origin backtest (supplementary)

```bash
python -m src.evaluate.rolling_backtest --config config/model.yaml
```
Writes `reports/rolling_backtest.csv` + `reports/figures/rolling_backtest_rmse.png`.
~Several minutes (6 folds × tuning); progress logged per fold.

### 11.6 Freeze the published model

```bash
python -m src.freeze_model --config config/model.yaml
```
Writes `models/final/{xgboost_final.json,xgboost_final.joblib,scaler.joblib,
feature_list.json,hyperparameters.json,SHA256SUMS}`.

### 11.7 Predict (frozen model)

```bash
python -m src.predict --date 2025-06-01
```
Outputs next-day (t+1) PM2.5 (µg/m³) + US-EPA AQI category. Refuses dates whose
45 features are unavailable.

### 11.8 Tests

```bash
pytest -q                      # reference-validation + predict unit tests
```

### 11.9 Verify the frozen-artifact checksums

```bash
cd models/final && sha256sum -c SHA256SUMS        # Linux/macOS
```
Cross-platform (Python):
```bash
python - <<'PY'
import hashlib, pathlib
base = pathlib.Path("models/final")
for line in (base / "SHA256SUMS").read_text().splitlines():
    want, name = line.split("  ", 1)
    p = (base / name).resolve()
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    print("OK " if got == want else "FAIL", name)
PY
```

### Explore it visually
There is also a point-and-click demo app — see [Part 10](#part-10--the-demo-app-streamlit-page-by-page)
(`streamlit run app/streamlit_app.py`).

### Notebooks

- `notebooks/01_data_acquisition.ipynb` — ingest + validation cells.
- `notebooks/02_modeling.ipynb` — full modeling, identical code path to
  `python -m src.run_experiment`; re-execute top-to-bottom to regenerate.
