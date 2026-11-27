# Dhaka next-day PM2.5 forecasting: a single-monitor benchmark (v2.0-single-station)

This repository holds the frozen code, pinned data, checksummed artifacts and revision analyses behind the manuscript *"Next-day PM2.5 forecasting at a single reference-grade monitor in Dhaka, Bangladesh: a provenance-audited benchmark of persistence, linear and tree-based models"* (under review at *Air Quality, Atmosphere & Health*). Author: Hrishin Debnath.

**What it shows.** On one reference-grade monitor (OpenAQ sensor 24434), a pre-selected XGBoost model forecasts tomorrow's daily-mean PM2.5 with RMSE **30.80 ug/m3** (R2 0.795) on a 12-month holdout, against **36.02** for persistence. Regularized linear regression and random forest do equally well. Recent PM2.5 and the calendar alone reach the same error; ERA5 reanalysis and FIRMS fire inputs add nothing measurable to them. This is a hindcast benchmark, not an operational forecast.

## Results on the chronological 12-month holdout (n = 219)

Persistence means "tomorrow equals today". DM p is a two-sided Diebold-Mariano test on squared error.

| Model | RMSE (ug/m3) | R2 | dRMSE vs persistence | DM p vs persistence | DM p, XGBoost vs model |
|---|---|---|---|---|---|
| Lasso (45 features) | 30.56 | 0.798 | -5.46 | 0.021 | 0.84 |
| Random forest | 30.64 | 0.797 | -5.38 | 0.034 | 0.83 |
| **XGBoost (pre-selected, frozen)** | **30.80** | **0.795** | **-5.22** | **0.022** | - |
| Ridge (45 features) | 31.30 | 0.789 | -4.73 | 0.052 | 0.68 |
| LightGBM | 31.52 | 0.785 | -4.50 | 0.063 | 0.23 |
| OLS on five PM2.5 lags | 32.33 | 0.774 | -3.70 | 0.039 | 0.32 |
| Persistence | 36.02 | 0.720 | - | - | 0.022 |

Read these numbers with the following results.

- **XGBoost is not better than linear regression.** No difference between XGBoost and lasso, ridge or random forest is significant. With a Holm correction across the three tree models, the XGBoost p-value against persistence is 0.065.
- **The advantage over persistence is specific to the last year.** In the four rolling-origin folds that do not overlap the holdout (test years 2020-2023, n = 748), XGBoost does not beat persistence (29.23 vs 29.73, p = 0.73). A regression on five PM2.5 lags does (27.56, p = 0.017).
- **ERA5 and FIRMS add no measurable skill.** PM2.5 history plus calendar (14 features) gives 30.33. Removing all ERA5 features gives 31.45 (p = 0.59 vs the full model); removing FIRMS gives 29.90 (p = 0.014, better). ERA5 carries 56% of the SHAP attribution, so SHAP share is not skill.
- **The worst days are underpredicted.** All 11 holdout days above 240 ug/m3 are underpredicted (mean bias -63). Category accuracy is 0.658 for XGBoost and 0.671 for persistence.
- **Hindcast only.** ERA5 is released about five days late, the lag-0 FIRMS features extend about six hours past the end of day t, and the monitor's public record ended on 24 March 2025.
- **Data screening.** The frozen model was trained with 43 hourly fault-like values (>= 900 ug/m3) in place. Masking them gives 29.89; this is reported as a sensitivity result only.

Ground truth: OpenAQ sensor 24434 (location 8415, provider AirNow), the reference-grade monitor at the US diplomatic post in Dhaka. All results apply to that location only.

## Why a single monitor

A query for PM2.5 within 25 km of the city centre returns 22 OpenAQ locations and 20 PM2.5 sensors with data. Two are reference monitors; 18 are low-cost sensors whose data begin on or after 28 October 2024. Sensor 4680 contributes only in 2016, and sensor 24434 is the only sensor in the pooled hourly mean from 2017 to 2023 (100% of hours); its share falls to 81.2% in 2024, 31.3% in 2025 and 0% in 2026. The target and all its lags use sensor 24434 only. Details: `approach2/reports/revision/r16_pooling_from_raw.json`.

## What is in the repository

- `approach2/` - the main pipeline: `src/`, `config/`, `tests/`, `reports/`, `models/final/` (frozen model, feature list, hyperparameters, **SHA256SUMS**) and the pinned dataset `data/processed/dhaka_aq_master_singlestation.parquet`.
- `approach2/reports/revision/` - every analysis added in the revision (benchmarks, ablation, rolling folds, significance, extremes, group SHAP, common-sample replication, pooling audit, feature timing, figures), with scripts, outputs and a README.
- `approach1/` - a separately written second pipeline on the same record (Optuna-tuned LightGBM, Bi-LSTM benchmark). It is a code replication, not an external validation. On the 218 holdout days shared with `approach2`, its LightGBM scores 30.38 and XGBoost 30.76 (p = 0.74 between them).
- `data_collection/` - the two upstream data-collection notebooks and their manifests.

The raw per-sensor OpenAQ downloads are not redistributed.

## Reproduce

Verify the checksums first, then run. The frozen model was trained with **xgboost 3.2.0**; other versions do not reproduce 30.80 exactly.

```bash
cd approach2
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt "xgboost==3.2.0"
(cd models/final && sha256sum -c SHA256SUMS)   # verify the frozen artifacts
PYTHONPATH=reports/revision python reports/revision/r00_reproduce_frozen.py   # 30.80, 0.795, n = 219
PYTHONPATH=reports/revision python reports/revision/r03_benchmarks.py          # benchmark table
```

`python -m src.run_experiment` retrains the original pipeline and rewrites `reports/` and `models/`; use it only in a copy if you want to keep the frozen outputs untouched. scikit-learn and LightGBM versions were not pinned, so random forest and LightGBM can differ in the second decimal (30.66 and 31.69 in a re-run).

## Data and code licensing

All inputs are open data: OpenAQ, ERA5 (Copernicus / Open-Meteo) and NASA FIRMS. The redistributed processed tables stay subject to those providers' terms. The code is MIT-licensed (see LICENSE).
