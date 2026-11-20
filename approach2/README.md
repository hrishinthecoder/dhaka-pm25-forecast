# Dhaka Next-Day PM2.5 Forecasting (main pipeline)

This pipeline forecasts the **next-day (t+1) 24-hour mean PM2.5** at one Dhaka
reference-grade monitor, **OpenAQ sensor 24434 (location 8415, provider AirNow; US
diplomatic post, about 2 km from the centre)**, from day-*t* information: PM2.5
history, ERA5 reanalysis meteorology, NASA FIRMS fire activity and calendar features.

On the held-out test window (the last 12 months of available pairs, n = 219; the sensor
record ends 2025-03) the frozen, pre-selected XGBoost model scores **RMSE 30.8
[25.9, 36.1] ug/m3**, against 36.0 for persistence (dRMSE -5.22 [-9.49, -0.77],
Diebold-Mariano p = 0.022; 0.065 after Holm correction across the three tree models).

The revision analyses in `reports/revision/` qualify this result:

- Lasso (30.56), random forest (30.64) and ridge (31.30) are statistically
  indistinguishable from XGBoost.
- PM2.5 history plus calendar gives 30.33. Removing ERA5 does not change the error significantly;
  removing FIRMS lowers it to 29.90.
- In the four rolling folds that do not overlap the holdout, XGBoost does not beat
  persistence (pooled p = 0.73); a five-lag linear regression does.
- All holdout days above 240 ug/m3 are underpredicted, and the model does not beat
  persistence on AQI category accuracy (0.658 vs 0.671).

> **Hindcast benchmark, not an operational advisory.** ERA5 is released about five days
> late, the lag-0 FIRMS features extend about six hours past the end of day *t*, and the
> monitor's public record ended on 24 March 2025. Single monitor; no citywide claim.

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
| `reports/revision/` | Revision analyses: benchmarks, ablation, fold dates, extremes, group SHAP, pooling audit, feature timing, figures (see its README). |
| `docs/` | `MASTER_DOCUMENTATION.md` (full guide: methods, results, model card, reproducibility). |
| `tests/` | `test_against_reference.py`, `test_predict.py` (7 tests; 4 are skipped when the reference file is absent). |
| `logs/` | Run logs. |

## Data licensing

The **data** are governed by their providers' terms: **OpenAQ** (CC BY 4.0),
**ERA5 / Copernicus CDS** (Copernicus licence), and **NASA FIRMS** (NASA data policy).
Comply with each source's licence when redistributing data or derived products.

## Documentation & links

- Full guide (beginner-friendly, all methods + results): [docs/MASTER_DOCUMENTATION.md](docs/MASTER_DOCUMENTATION.md)
- Demo app notes: [app/README.md](app/README.md)
- Frozen-artifact checksums: [models/final/SHA256SUMS](models/final/SHA256SUMS)

Status: **Model frozen (v2.0, single-station sensor 24434). Manuscript in revision; frozen artifacts unchanged.**
