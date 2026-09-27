
# Dhaka next-day PM2.5 forecasting (v2.0-single-station)

**Introduction:** an XGBoost model forecasts tomorrow's daily-mean PM2.5 in Dhaka with RMSE **30.80 ug/m3** (R2 0.795). That beats persistence by **5.22 ug/m3** (Diebold-Mariano p = **0.022**). This repository holds the frozen code, pinned data, and checksummed artifacts behind the manuscript *"Next-day PM2.5 forecasting in Dhaka, Bangladesh, from a single reference-grade monitor"*. The paper is under review at *Air Quality, Atmosphere & Health*. Author: Hrishin Debnath.

## Results on the chronological 12-month holdout

Persistence means "tomorrow equals today". It is the baseline any forecast must beat.

| Model | RMSE (ug/m3) | R2 | dRMSE vs persistence | DM p |
|---|---|---|---|---|
| Persistence | 36.02 | 0.720 | - | - |
| **XGBoost (primary, Approach 2)** | **30.80** | **0.795** | **-5.22** | **0.022** |
| LightGBM (Approach 1 replication) | 30.25 | 0.809 | -5.33* | n/t |
| Bi-LSTM+attention (Approach 1) | 30.73 | 0.803 | -4.85* | n/t |

\* against Approach 1's own persistence baseline (35.58, n = 236 holdout). The two approaches were built independently and use slightly different day masks, so their deltas are not cross-comparable.

Ground truth: OpenAQ sensor 24434, the reference-grade monitor at the US Diplomatic Post, Dhaka. In a rolling-origin backtest, XGBoost beats persistence in **3 of 6** yearly folds (2022, 2024, 2025).

## What is in the repository

- `approach2/` - the primary XGBoost pipeline: `src/`, `config/`, `tests/`, `reports/` (all metrics, figures, and `RESULTS_FACTS.md`), `models/final/` (frozen model, feature manifest, hyperparameters, **SHA256SUMS**), and the pinned dataset `data/processed/dhaka_aq_master_singlestation.parquet`.
- `approach1/` - the independent replication (Optuna-tuned LightGBM) and the Bi-LSTM benchmark: `src/`, `reports/phase*.md`, `figures/`, and the frozen LightGBM artifact.
- `data_collection/` - the two upstream data-collection notebooks and their manifests.

## Reproduce the results

Verify the checksums first. They prove your copy matches the artifacts the paper reports, byte for byte. Then retrain from the pinned parquet.

```bash
cd approach2
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # keys needed only if re-pulling raw data
sha256sum -c models/final/SHA256SUMS           # verify the frozen artifacts first
python -m src.run_experiment                   # retrains from the pinned parquet
```

Every number in the paper traces to `approach2/reports/RESULTS_FACTS.md` and `approach2/models/final/SHA256SUMS` (tag `v2.0-single-station`). Load the model with joblib first; fall back to `XGBRegressor.load_model()` on the JSON.

## Data and code licensing

All inputs are open data: OpenAQ, ERA5 (Copernicus / Open-Meteo), and NASA FIRMS. The redistributed processed tables stay subject to those providers' terms. The code is MIT-licensed (see LICENSE).
