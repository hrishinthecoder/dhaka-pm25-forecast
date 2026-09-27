# Streamlit Proof-of-Concept App

A local **display layer** over the frozen Dhaka next-day PM2.5 model — built to
demonstrate the model to collaborators (DoE / BAPA / BUET). It is **not** an
operational forecasting service, and the app says so on every screen.

## Launch

From the repository root (with the project environment active):

```bash
streamlit run app/streamlit_app.py
```

Starts in seconds (model/scaler cached with `st.cache_resource`; parquet and the
recent-forecasts computation cached with `st.cache_data`).

## What it shows

- **Forecast** — pick a day-*t* (bounded to dates with complete features); get the
  predicted next-day 24-h mean PM2.5, the US-EPA AQI band (standard color + one-line
  health note), and — when the record contains it — the observed next-day value with
  the absolute error. Unavailable dates show the model's refusal message, never a crash.
- **Model performance** — headline metrics with bootstrap CIs (`reports/metrics.csv`),
  paired significance vs persistence (`reports/significance_vs_persistence.csv`),
  seasonal skill (`reports/seasonal_metrics.csv`), and the rolling-backtest figure.
  All numbers are read from the report files — none are hard-coded.
- **Recent forecasts** — last 90 available days of observed daily-mean PM2.5 with the
  model's next-day predictions overlaid and EPA band boundaries as reference lines.
- **About / model card** — renders `docs/MASTER_DOCUMENTATION.md` (overview, model card,
  methods, results, reproducibility), plus a data provenance line.

## Guarantees / scope

- **Read-only and frozen-only.** Loads exclusively from `models/final/`
  (`xgboost_final.json/.joblib`, `scaler.joblib`, `feature_list.json`,
  `hyperparameters.json`) and reads `data/processed/dhaka_aq_master_rebuilt.parquet`.
- **Checksum gate.** At startup it verifies every artifact against
  `models/final/SHA256SUMS`. On any mismatch it shows a red banner and **refuses to
  predict**.
- **Shared prediction path.** All inference goes through `src/predict.py`
  (`_predict_from_frame` / `predict_series`) — no duplicated feature engineering. A
  date forecast in the app matches `python -m src.predict --date <same date>` exactly.
- **No** retraining, retuning, model switching, file uploads into the model, or live
  API pulls. Nothing under `models/final/` or the modeling code is modified.

## Out of scope for this phase

**Public deployment is out of scope.** This is a local demonstration only. Operational
use would require substituting NWP forecast inputs for the ERA5 reanalysis predictors
and is not provided here.
