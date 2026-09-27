# Model card — persistence

**Type:** baseline  
**Task:** next-day (t+1) 24-h mean PM2.5 (µg/m³), Option A  
**Best model:** no  
**Seed:** 42

## Test-set performance (point + 95% bootstrap CI, n_boot=1000)

| metric | value | 95% CI |
|---|---|---|
| RMSE | 36.025 | [30.382, 41.615] |
| MAE | 23.343 | [19.697, 27.222] |
| R² | 0.720 | [0.614, 0.793] |
| bias | 0.596 | [-3.984, 5.350] |
| n (test) | 219 | — |

## Features

45 day-t features (no t+1, no same-station o3/pm10, no satellite PM2.5). See reports/feature_availability.csv.

## Notes
- ERA5 = reanalysis used as a day-t predictor proxy.
- BLH for the 2024 H1 gap is sourced from native ERA5 (Copernicus CDS); `era5_blh_imputed_day` is now constant 0 (no imputed hours remain). Fallback when CDS is unavailable: train-years-only (<=2023) month×hour climatology.
