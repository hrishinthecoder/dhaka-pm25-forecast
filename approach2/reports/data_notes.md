# Data Notes — Notebook 02 Modeling (Option A, next-day daily-mean PM2.5)

Source of truth: `config/model.yaml`. Input:
`data/processed/dhaka_aq_master_singlestation.parquet` (87,672 hourly rows × 27 cols).
All imputation / exclusion decisions below.

## Target & framing — SINGLE STATION
- **Target** = next-day (t+1) **24-h mean** `openaq_pm25` (µg/m³) at **one reference
  monitor — OpenAQ sensor 24434 (US Diplomatic Post, Dhaka, ~2 km from centre)**. The
  hourly `openaq_pm25` column is that single sensor only, **not** a cross-station mean.
  Built by `fusion/swap_target.py`; rule documented in `src/ingest/merge.py`.
- Secondary: next-day AQI category (US EPA 2024 24-h PM2.5 breakpoints).
- A daily value is **valid** only with ≥18 hourly obs (`coverage.min_hourly_obs_per_day`).
  A (t → t+1) pair is kept only if **both** days are valid (`require_both_days`).
- This is a **temporal** forecast at a single location — never spatial interpolation or
  sensor substitution.

## Temporal split — anchored holdout (no shuffle, ever)
Sensor 24434 ends **2025-03-24**, so a fixed "≥2025" calendar split would leave too few
test days. The holdout is instead the **last 12 months of *available* labeled pairs**,
anchored to the last valid day-t (`split.mode: anchored_holdout`, `holdout_months: 12`).

| split | rule | model rows |
|-------|------|-----------|
| pre-holdout (train + CV) | day-t < 2024-03-23 | 1,341 |
| holdout (test) | day-t 2024-03-23 → 2025-03-23 (last 12 mo) | 219 |

Total labeled (t→t+1) pairs that reach the model: **1,560**. Hyperparameters are tuned by
`TimeSeriesSplit(5)` over the **whole pre-holdout window** and the final model is **refit on
that same pre-holdout window**; the holdout is scored **once**. Model rows < raw valid pairs
because rows with any missing **feature** (a PM2.5 lag/rolling value on a day near a coverage
gap) are dropped, never imputed — we never fabricate predictor history.

- **2018 station outage**: sparse valid pairs in 2018 (Aug–Nov gap, sensor 24434). Year is
  **kept, not dropped** — removing it would bias the dry-season climatology. Flagged per audit.

## Boundary-layer height (BLH) — sourced from CDS (ERA5 native archive)
`era5_boundary_layer_height` was missing for **2024-01-01 → 2024-06-30** (4,368 h), but
this was **a gap in the Open-Meteo ERA5 *mirror*, not a source gap** — the variable exists
in the native CDS archive. (The earlier note claiming a "genuine ERA5/Open-Meteo upstream
data gap" was wrong: ERA5 BLH is fully served by the Copernicus Climate Data Store.)
- **BLH for Jan–Jun 2024 is sourced directly from CDS (ERA5 native archive)** via
  `src/ingest/cds_blh.py` (dataset `reanalysis-era5-single-levels`, variable
  `boundary_layer_height`, nearest gridpoint to 23.8103 N / 90.4125 E, CDS-Beta endpoint),
  patched into the master by `src/ingest/patch_blh.py`. Real-reanalysis mean over the window
  = **492.85 m** (range 10.6–2884.9 m). Those 4,368 hours are now real data →
  `era5_blh_imputed = 0` for them (**0 imputed hours remain** in the master).
- **Fallback (PATH B)**, used only when CDS credentials are absent: month × hour BLH
  climatology computed from **training years (≤2023) only** — never later years (which would
  leak future information backward) — flagged via `era5_blh_imputed`. Implemented in
  `src/ingest/merge.py::_impute_blh`.
- In modeling, BLH is **used as a feature** (mean/min/max daily aggregates). The daily flag
  `era5_blh_imputed_day` is **dropped from the feature matrix** (constant 0 after the CDS
  patch — a dead, zero-variance column).
- The CDS-sourced window mean (492.85 m) differs from the prior climatological imputation
  (496.9 m) by <1%, retrospectively confirming the imputation introduced no material distortion.

## Feature set (all day-t; see `reports/feature_availability.csv`)
- **PM2.5 history**: `pm25_t`, lags {1,2,3,7}, rolling mean+std {3,7} (day t and earlier) —
  now all from sensor 24434.
- **ERA5 (day-t reanalysis)**: mean/min/max of temperature, RH, dew point, surface
  pressure, cloud cover, wind speed, BLH; **daily sum** of precipitation and shortwave
  radiation; **wind_u_mean / wind_v_mean** vector decomposition from speed + direction.
- **FIRMS** fire activity: `firms_count`, `firms_frp_sum` at lags {0,1,2}.
- **Calendar**: day_of_week, month, is_weekend, is_holiday, is_ramadan (day-level).
- Total: **45** model features.

### Excluded as predictors (leakage control)
- **`openaq_o3`, `openaq_pm10`** — same monitoring station as the target; not guaranteed
  available at forecast time. Excluded (config `leakage_guards`), enforced in
  `src/features/daily.py` (`LEAKAGE_EXCLUDED`).
- **Satellite PM2.5** (`omaq_*`, `acag_*`) — same-period satellite estimate of the target;
  excluded (none present in the rebuilt master anyway; guarded by name).
- No t+1 fields anywhere.

### ERA5 availability caveat (honest framing)
ERA5 is **reanalysis**, used here as a day-t predictor proxy. An operational next-day
system would substitute a numerical-weather-prediction (NWP) **forecast** for day t+1.
We follow the standard reanalysis-as-predictor convention for the retrospective study and
document the substitution required for deployment.

## Reproducibility
- Single seed (42) from `config/model.yaml` everywhere; no shuffle; scalers fit on the
  pre-holdout window only.
- **Selection rule:** hyperparameters selected via rolling-origin `TimeSeriesSplit(5)` over
  the **whole pre-holdout window** (day-t < 2024-03-23); final models **refit on that same
  window**. The **12-month holdout (2024-03-23 → 2025-03-23) was untouched by any selection
  step.** Primary model chosen by **lowest CV-RMSE among the tuned models** (XGBoost 31.14 <
  LightGBM 32.19); RandomForest is a fixed (untuned) baseline. RF fixed (n_estimators=500).
- One command reproduces every artifact:
  `python -m src.run_experiment --config config/model.yaml`

## Headline results (test, n=219; 95% bootstrap CI, n_boot=1000)
| model | RMSE | R² | MAE | beats persistence? (paired) |
|-------|------|----|-----|------------------|
| **XGBoost (primary)** | 30.80 [25.93, 36.14] | 0.795 [0.732, 0.841] | 21.42 | yes — ΔRMSE CI excludes 0, DM p=0.022 |
| Random Forest | 30.64 [25.49, 36.17] | 0.797 | 21.19 | yes — ΔRMSE CI excludes 0, DM p=0.034 |
| LightGBM | 31.52 [26.43, 37.07] | 0.785 | 21.79 | borderline — see below |
| persistence (baseline) | 36.02 [30.38, 41.62] | 0.720 | 23.34 | — |
| climatology (baseline) | 37.95 [32.21, 43.95] | 0.689 | 24.79 | — |

RandomForest has a marginally lower **point** test-RMSE (30.64 vs 30.80) but it is the
**untuned** baseline with no CV score — selecting it as primary would be selecting on the
test set. The pre-registered CV rule picks **XGBoost**. The two are tied within noise.

These **marginal** CIs overlap (persistence sits inside the ML models' RMSE CIs), so they
cannot establish the comparison. The **paired** tests below can
(`reports/significance_vs_persistence.csv`).

### Significance vs persistence (paired bootstrap ΔMAE/ΔRMSE + Diebold–Mariano, h=1, HLN-corrected)
Δ = model − persistence (negative ⇒ model better). Model beats persistence iff the Δ CI excludes 0.
- **XGBoost vs persistence:** ΔRMSE = −5.22 [−9.49, −0.77] (CI excludes 0 ⇒ beats on RMSE);
  ΔMAE = −1.93 [−4.49, 0.50] (CI spans 0); DM p = **0.022**.
- **Random Forest vs persistence:** ΔRMSE = −5.38 [−10.18, −0.55] (CI excludes 0); ΔMAE =
  −2.15 [−4.67, 0.31]; DM p = 0.034.
- **LightGBM vs persistence:** ΔRMSE = −4.50 [−9.19, −0.06] (CI **marginally** excludes 0) but
  DM p = **0.063** > 0.05 ⇒ **does not robustly separate** from persistence.
- **Climatology vs persistence (sanity row):** ΔRMSE = +1.92 [−3.91, 7.94] ⇒ climatology is
  *worse* than persistence, as expected.

Honest read: **XGBoost and RF** separate from persistence on the paired RMSE difference;
**LightGBM is borderline and does not**. The gain over persistence is concentrated in **large
errors** (ΔRMSE significant, ΔMAE not) — i.e. the model helps most on high-pollution episode
days. Real but modest, on a thin single-station test window.

### Rolling-origin multi-year backtest (`reports/rolling_backtest.csv`, `figures/rolling_backtest_rmse.png`)
Six expanding-window folds (train ≤ Dec 31 y, test y+1; same builder/grids/seed; tune
TimeSeriesSplit(5) on each fold's train window, refit on it). The final fold (test 2025) has
only **n=75** because sensor 24434 ends 2025-03-24.

XGBoost vs persistence, **paired ΔRMSE (CI excludes 0 ⇒ beats persistence)** — reported verbatim:
| test year | n | XGB RMSE | persist RMSE | ΔRMSE [95% CI] | DM p | beats? |
|---|---|---|---|---|---|---|
| 2020 | 198 | 28.02 | 26.28 | **+1.74** [−2.50, 6.09] | 0.413 | no (marginally worse) |
| 2021 | 164 | 34.44 | 36.25 | −1.81 [−9.10, 5.73] | 0.636 | no (CI spans 0) |
| 2022 | 175 | 27.87 | 33.34 | −5.47 [−10.38, −0.97] | 0.048 | **yes** |
| 2023 | 211 | 26.95 | 23.27 | **+3.69** [−0.85, 8.38] | 0.139 | no (worse) |
| 2024 | 206 | 25.78 | 29.45 | −3.67 [−6.24, −1.23] | 0.012 | **yes** |
| 2025 | 75 | 46.79 | 55.62 | −8.83 [−17.11, −1.43] | 0.035 | **yes** (n=75) |

**XGBoost beats persistence on paired ΔRMSE in 3 of 6 folds (2022, 2024, 2025).** In 2021 the
CI spans 0; in **2020 and 2023 XGBoost is marginally worse than persistence**. This is a
finding, not a bug: next-day skill over persistence is **real but not uniform across the folds,
and not a recency or training-length trend** — 2022 wins yet 2023 loses. The gains concentrate
on high-error / episode days (ΔRMSE separates from persistence while ΔMAE does not). Reported
without aggregating away the failures.

By season (XGBoost holdout, n=219): monsoon RMSE 11.43 (R² 0.21, n=81), transition 30.03
(R² 0.19, n=86), winter-dry 47.97 (R² 0.32, n=52) — winter is hardest (high-pollution
variance). AQI classification: accuracy 0.658, macro-F1 0.503.
</content>
</invoke>
