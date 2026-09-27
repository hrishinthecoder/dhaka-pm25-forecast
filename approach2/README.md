# Dhaka Next-Day PM2.5 Forecasting

This project forecasts the **next-day (t+1) 24-hour mean PM2.5** at a single Dhaka
reference-grade station — **OpenAQ sensor 24434 (US Diplomatic Post, ~2 km from
centre)** — from day-*t* information — ERA5 reanalysis meteorology (via
Open-Meteo, with the Jan–Jun 2024 boundary-layer-height window pulled from the native
Copernicus CDS archive), NASA FIRMS fire activity, and calendar proxies — using a
gradient-boosted tree model (XGBoost). On the held-out test window — the last 12 months
of available labeled pairs (n=219; sensor 24434 ends 2025-03) — the frozen model scores
**RMSE 30.8 [25.9, 36.1] µg/m³** and **beats a persistence (same-as-yesterday) baseline
on the paired ΔRMSE difference** (−5.22 [−9.49, −0.77], Diebold–Mariano p=0.022); the
advantage concentrates in large errors (ΔRMSE significant, ΔMAE not). A rolling backtest
shows this edge in 3 of 6 yearly folds (2022, 2024, 2025). Random Forest separates from
persistence similarly; **LightGBM does not robustly separate from persistence (DM
p=0.063)**.

> **Research proof of concept — not an operational advisory.** It uses ERA5
> *reanalysis* as a day-*t* proxy; an operational next-day system would require a
> numerical-weather-prediction (NWP) *forecast* for day t+1. Single station; not a
> substitute for regulatory monitoring.

## Quickstart

```bash
# 0. environment
python -m venv .venv && source .venv/bin/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# keys: OPENAQ_API_KEY + FIRMS_MAP_KEY in .env; Copernicus CDS token in ~/.cdsapirc.
# NOTE: exact checksum reproduction REQUIRES CDS credentials (the BLH gap is pulled
# from the native ERA5/CDS archive). Without them the run falls back to a train-only
# climatology and the parquet will not match models/final/SHA256SUMS.

# 1. ingest -> data/processed/ (auto-patches the H1-2024 BLH gap)
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 notebooks/01_data_acquisition.ipynb
# 2. headline experiment + all reports
python -m src.run_experiment --config config/model.yaml
# 3. rolling-origin backtest
python -m src.evaluate.rolling_backtest --config config/model.yaml
# 4. freeze the published model + checksums
python -m src.freeze_model --config config/model.yaml
# 5. predict one day
python -m src.predict --date 2025-06-01
# 6. tests
pytest -q
```

A point-and-click demo (read-only over the frozen model): `streamlit run app/streamlit_app.py`.

## Repository map

| Path | Contents |
|---|---|
| `config/` | Run configuration (`sources.yaml`, `model.yaml`, resolved stations); seed 42. |
| `src/` | Pipeline code: `ingest/`, `features/`, `models/`, `evaluate/`, plus `run_experiment.py`, `freeze_model.py`, `predict.py`. |
| `app/` | Streamlit proof-of-concept display app (read-only). |
| `notebooks/` | `01_data_acquisition.ipynb` (ingest), `02_modeling.ipynb` (modeling). |
| `data/` | `raw/` caches, `interim/` (incl. the CDS BLH NetCDF), `processed/` master, `reference/` validation file. |
| `models/` | `final/` frozen artifacts + `SHA256SUMS`; other model files are gitignored. |
| `reports/` | Metrics, significance, backtest, seasonal, SHAP, figures, model cards. |
| `docs/` | `MASTER_DOCUMENTATION.md` (full guide: methods, results, model card, reproducibility). |
| `tests/` | `test_against_reference.py`, `test_predict.py` (7 tests). |
| `logs/` | Run logs. |

## Data licensing

The **data** are governed by their providers' terms: **OpenAQ** (CC BY 4.0),
**ERA5 / Copernicus CDS** (Copernicus licence), and **NASA FIRMS** (NASA data policy).
Comply with each source's licence when redistributing data or derived products.

## Documentation & links

- Full guide (beginner-friendly, all methods + results): [docs/MASTER_DOCUMENTATION.md](docs/MASTER_DOCUMENTATION.md)
- Demo app notes: [app/README.md](app/README.md)
- Frozen-artifact checksums: [models/final/SHA256SUMS](models/final/SHA256SUMS)

Status: **Model frozen (v2.0, single-station sensor 24434). Manuscript in preparation.**
