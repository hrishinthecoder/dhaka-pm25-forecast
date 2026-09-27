# Phase 8 — Ablations (which signals carry the forecast)

_Generated 2026-06-10._ LightGBM with FIXED Phase-5 best params; only the feature set varies. Same expanding walk-forward folds + untouched hold-out. Baseline bar: persistence CV RMSE 33.12, hold-out 35.58.

| Experiment | #feat | CV RMSE | CV R² | HO RMSE | skill vs persist (CV) |
|---|---|---|---|---|---|
| met_only | 25 | 35.30 ± 6.75 | 0.722 | 34.21 | -0.066 |
| met_calendar | 33 | 34.91 ± 6.73 | 0.728 | 32.03 | -0.054 |
| met_calendar_lags | 49 | 30.24 ± 7.55 | 0.792 | 30.20 | +0.087 |
| core | 51 | 30.31 ± 7.45 | 0.791 | 30.34 | +0.085 |
| core_plus_chemistry | 55 | 30.31 ± 7.45 | 0.791 | 30.18 | +0.085 |
| core_plus_satellite_anchor | 54 | 30.57 ± 7.27 | 0.788 | 30.28 | +0.077 |
| core_plus_cams_pm25_CAVEATED | 52 | 30.31 ± 7.45 | 0.791 | 30.34 | +0.085 |

## The `omaq_pm2_5` ablation (circularity caveat)

- core CV RMSE **30.31** vs core+omaq_pm2_5 CV RMSE **30.31** (Δ +0.00; positive Δ = the CAMS PM2.5 feature lowers error).

> **Circularity caveat:** `omaq_pm2_5` is a CAMS/Open-Meteo MODEL OUTPUT of PM2.5 (hourly r≈0.754 with measured PM2.5, ~86% missing). Predicting *measured* PM2.5 from a model's PM2.5 estimate is near-circular, so any gain it brings is **not** evidence of a better forecaster. It is reported here for disclosure only and is EXCLUDED from the default/core model.


## Interpretation
- Marginal group contributions are read by walking met_only → +calendar → +lags/rolling → +fire (core). The largest CV-RMSE drop marks the signal that carries the forecast.