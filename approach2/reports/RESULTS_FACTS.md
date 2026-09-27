# RESULTS_FACTS — writer hand-off (v2.0-single-station)

> **This is a FACTS-AND-TABLES sheet, NOT a Results section and NOT prose.**
> Every number traces to a file in `reports/` or `models/final/`. The manuscript
> team writes the Results prose from this sheet; do not lift sentences from here as-is.
> Source files: `reports/metrics.csv`, `reports/significance_vs_persistence.csv`,
> `reports/rolling_backtest.csv`, `reports/seasonal_metrics.csv`,
> `reports/classification_metrics.json`, `models/final/SHA256SUMS`, `config/model.yaml`.
> Numbers are read verbatim; CIs are 95% paired bootstrap (n_boot=1000); DM = Diebold–Mariano
> HLN-corrected, h=1. Model is frozen (`v2.0-single-station`) — these numbers do not change.

---

## A. Study setup

- **Target:** next-day (t→t+1) daily-mean PM2.5 (µg/m³) at a **single station** —
  OpenAQ sensor **24434**, "US Diplomatic Post, Dhaka (~2 km from centre)"
  (`config/model.yaml: target.sensor_id=24434, target.source=single_sensor`). NOT a
  cross-station pooled mean.
- **Inputs:** ERA5 reanalysis meteorology (day-t proxy), NASA FIRMS fire activity, calendar
  features. PM2.5 lags/rolling from day-t and earlier only. Leakage guards: no t+1 features,
  no same-period satellite PM2.5, no same-station o3/pm10 as predictors.
- **Split:** anchored holdout = last 12 months of *available* labeled pairs
  (`split.mode=anchored_holdout, holdout_months=12`); anchored to the sensor's last valid day
  because 24434 ends 2025-03-24.
- **Sample:** 1,560 valid (t→t+1) pairs total. **Train (pre-holdout) = 1,341; holdout = 219.**
  Holdout day-t window 2024-03-23 → 2025-03-23. Seasonal split of the 219 holdout pairs:
  winter-dry 52, transition 86, monsoon 81.
- **Frozen artifact:** XGBoost (`models/final/`), pinned to
  `data/processed/dhaka_aq_master_singlestation.parquet` via `models/final/SHA256SUMS`.

## B. Headline results — holdout (n=219)

| Model | RMSE [95% CI] | R² [95% CI] | MAE [95% CI] |
|---|---|---|---|
| persistence (baseline) | 36.02 [30.38, 41.62] | 0.720 [0.614, 0.793] | 23.34 [19.70, 27.22] |
| climatology (baseline) | 37.95 [32.21, 43.95] | 0.689 [0.604, 0.755] | 24.79 [21.24, 28.72] |
| RandomForest | 30.64 [25.49, 36.17] | 0.797 [0.743, 0.843] | 21.19 [18.27, 24.40] |
| **XGBoost (PRIMARY)** | **30.80 [25.93, 36.14]** | **0.795 [0.732, 0.841]** | **21.42 [18.57, 24.48]** |
| LightGBM | 31.52 [26.43, 37.07] | 0.785 [0.723, 0.836] | 21.79 [18.97, 24.95] |

- **XGBoost = primary** by the pre-registered rule: lowest tuned cross-validation RMSE among
  tuned models (selected before touching the holdout — not selected on test).
- RandomForest's holdout point RMSE is 0.16 lower than XGBoost's, but RF is untuned (no CV-RMSE
  selection score); their CIs overlap fully.

## C. Significance vs persistence — holdout (`significance_vs_persistence.csv`)

| Model | ΔRMSE [95% CI] | DM p-value | ΔMAE [95% CI] |
|---|---|---|---|
| RandomForest | −5.38 [−10.18, −0.55] | 0.034 | −2.15 [−4.67, +0.31] |
| **XGBoost** | **−5.22 [−9.49, −0.77]** | **0.022** | −1.93 [−4.49, +0.50] |
| LightGBM | −4.50 [−9.19, −0.06] | 0.063 | −1.55 [−4.32, +0.94] |
| climatology | +1.92 [−3.91, +7.94] | 0.553 | +1.45 [−2.03, +5.33] |

- ΔRMSE negative = better than persistence. XGBoost and RF ΔRMSE CIs exclude 0; LightGBM CI
  upper bound ≈ 0 and DM p ≈ 0.063 → **LightGBM does NOT robustly separate from persistence.**
- All three ML ΔMAE CIs include 0 → **no model robustly beats persistence on MAE** (skill is on
  RMSE, i.e. high-error days, not on the typical day).

## D. Rolling-origin backtest — XGBoost vs persistence (`rolling_backtest.csv`)

| Test year | n | ΔRMSE (XGB − persist) [95% CI] | DM p | vs persistence |
|---|---|---|---|---|
| 2020 | 198 | +1.74 [−2.50, +6.09] | 0.413 | lose |
| 2021 | 164 | −1.81 [−9.10, +5.73] | 0.636 | tie (CI spans 0) |
| 2022 | 175 | −5.47 [−10.38, −0.97] | 0.048 | **WIN** |
| 2023 | 211 | +3.69 [−0.85, +8.38] | 0.139 | lose |
| 2024 | 206 | −3.67 [−6.24, −1.23] | 0.012 | **WIN** |
| 2025 | 75 | −8.83 [−17.11, −1.43] | 0.035 | **WIN** (thin fold, n=75) |

- **XGBoost beats persistence (ΔRMSE CI excludes 0) in 3 of 6 folds** (2022, 2024, 2025).
- 2025 fold is thin (n=75) because sensor 24434 ends 2025-03-24 — high variance, flag in text.

## E. Seasonal + classification — holdout

Seasonal regression, XGBoost (`seasonal_metrics.csv`):

| Season | n | RMSE | MAE | R² |
|---|---|---|---|---|
| winter-dry | 52 | 47.97 | 35.64 | 0.322 |
| transition | 86 | 30.03 | 24.77 | 0.190 |
| monsoon | 81 | 11.43 | 8.72 | 0.207 |

AQI classification (`classification_metrics.json`): **accuracy 0.658, macro-F1 0.503, n=219.**

## F. Atomic factual statements (lift-ready, each tied to a source number)

1. On the 12-month holdout (n=219), XGBoost achieved RMSE = 30.80 µg/m³ [25.93, 36.14], lower than persistence (36.02) by ΔRMSE = −5.22 [−9.49, −0.77] (DM p = 0.022).
2. XGBoost holdout R² = 0.795 [0.732, 0.841]; MAE = 21.42 µg/m³ [18.57, 24.48].
3. The persistence baseline scored RMSE = 36.02 µg/m³ [30.38, 41.62], R² = 0.720, MAE = 23.34 on the same holdout.
4. Climatology did not beat persistence (ΔRMSE = +1.92 [−3.91, +7.94], DM p = 0.553).
5. RandomForest holdout RMSE = 30.64 µg/m³ [25.49, 36.17] (ΔRMSE = −5.38, DM p = 0.034); point RMSE 0.16 below XGBoost, CIs fully overlapping.
6. LightGBM holdout ΔRMSE vs persistence = −4.50 [−9.19, −0.06], DM p = 0.063 — does not robustly separate from persistence.
7. All three ML models' ΔMAE CIs include 0 (XGBoost −1.93 [−4.49, +0.50]); none robustly beats persistence on MAE.
8. In rolling-origin backtest, XGBoost beat persistence (ΔRMSE CI excluding 0) in 3 of 6 folds (test years 2022, 2024, 2025).
9. The 2025 backtest fold has n = 75 (sensor ends 2025-03-24); its ΔRMSE = −8.83 [−17.11, −1.43], DM p = 0.035.
10. Seasonal holdout RMSE: winter-dry 47.97 (R² 0.322, n=52), transition 30.03 (R² 0.190, n=86), monsoon 11.43 (R² 0.207, n=81) — winter-dry is the highest-error regime, monsoon the lowest.
11. AQI-category classification on the holdout: accuracy 0.658, macro-F1 0.503 (n=219).
12. The target is a single reference monitor (OpenAQ sensor 24434, US Diplomatic Post, ~2 km), not a cross-station mean; covariates may pool across stations, the target does not.
13. Training used 1,341 pre-holdout pairs; holdout used 219; 1,560 valid (t→t+1) pairs total.
14. XGBoost was selected as primary by lowest tuned cross-validation RMSE (pre-registered rule), before evaluation on the holdout.
15. The frozen XGBoost artifact is pinned by SHA256 to `dhaka_aq_master_singlestation.parquet` (`models/final/SHA256SUMS`).

## G. Framing rules for the writers (numbers-grounded may / must-not)

**MAY state:**
- The model beats the persistence baseline on an honest single station (XGBoost ΔRMSE −5.22, DM p = 0.022).
- Skill is concentrated on high-error / episode days: ΔRMSE separates from persistence, ΔMAE does not (all ΔMAE CIs include 0).
- Winter-dry is the hardest regime (RMSE 47.97), monsoon the easiest (RMSE 11.43).
- The pipeline is frozen and reproducible (SHA256-pinned data + frozen model, tag v2.0-single-station).

**MUST NOT state:**
- Uniform or all-year skill — it is 3 of 6 backtest folds.
- Any city-wide or spatial-coverage claim — single station (sensor 24434) only.
- That LightGBM beats persistence — it does not robustly (DM p ≈ 0.063).
- That the model replaces monitoring sensors.
