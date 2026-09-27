# Model card — climatology

**Type:** baseline  
**Task:** next-day (t+1) 24-h mean PM2.5 (µg/m³), Option A  
**Best model:** no  
**Seed:** 42

## Test-set performance (point + 95% bootstrap CI, n_boot=1000)

| metric | value | 95% CI |
|---|---|---|
| RMSE | 37.947 | [32.211, 43.947] |
| MAE | 24.792 | [21.238, 28.723] |
| R² | 0.689 | [0.604, 0.755] |
| bias | -2.241 | [-7.445, 3.019] |
| n (test) | 219 | — |

## Features

45 day-t features (no t+1, no same-station o3/pm10, no satellite PM2.5). See reports/feature_availability.csv.

## Notes
- ERA5 = reanalysis used as a day-t predictor proxy.
- BLH for the 2024 H1 gap is sourced from native ERA5 (Copernicus CDS); `era5_blh_imputed_day` is now constant 0 (no imputed hours remain). Fallback when CDS is unavailable: train-years-only (<=2023) month×hour climatology.
