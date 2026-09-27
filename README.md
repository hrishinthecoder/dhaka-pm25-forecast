# Dhaka next-day PM2.5 forecasting (v2.0-single-station)

Frozen code, data, and artifacts for the manuscript *"Next-day PM2.5 forecasting in
Dhaka, Bangladesh, from a single reference-grade monitor"* (under review, *Air
Quality, Atmosphere & Health*). Author: Hrishin Debnath.

## Headline results (frozen, chronological 12-month holdout)

| Model | RMSE (ug/m3) | R2 | dRMSE vs persistence | DM p |
|---|---|---|---|---|
| Persistence | 36.02 | 0.720 | - | - |
| **XGBoost (primary, Approach 2)** | **30.80** | **0.795** | **-5.22** | **0.022** |
| LightGBM (Approach 1 replication) | 30.25 | 0.809 | -5.33* | n/t |
| Bi-LSTM+attention (Approach 1) | 30.73 | 0.803 | -4.85* | n/t |

\* vs Approach 1's own persistence baseline (35.58, n = 236 holdout). Target:
OpenAQ sensor 24434 (US Diplomatic Post, Dhaka). Rolling-origin backtest:
XGBoost beats persistence in 3 of 6 folds (2022, 2024, 2025).

## Layout

- `approach2/` - primary pipeline (XGBoost): `src/`, `config/`, `tests/`,
  `reports/` (all metrics + figures + RESULTS_FACTS.md), `models/final/`
  (frozen model, feature manifest, hyperparameters, **SHA256SUMS**), and the
  pinned dataset `data/processed/dhaka_aq_master_singlestation.parquet`.
- `approach1/` - independent replication (Optuna LightGBM) and the Bi-LSTM
  benchmark: `src/`, `reports/phase*.md`, `figures/`, frozen LightGBM artifact.
- `data_collection/` - the two upstream data-collection notebooks/manifests.

## Reproduce

```bash
cd approach2
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # keys needed only if re-pulling raw data
sha256sum -c models/final/SHA256SUMS           # verify the frozen artifacts first
python -m src.run_experiment                   # retrains from the pinned parquet
```

Every number in the paper traces to `approach2/reports/RESULTS_FACTS.md` and
`approach2/models/final/SHA256SUMS` (tag `v2.0-single-station`). Model load
order: joblib first, `XGBRegressor.load_model()` on the JSON as fallback.

## Data licensing

Inputs are open data from OpenAQ, ERA5 (Copernicus / Open-Meteo), and NASA
FIRMS; use of the redistributed processed tables is subject to those providers'
terms. Code is MIT-licensed (see LICENSE).
