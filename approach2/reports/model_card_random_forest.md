# Model card — random_forest

**Type:** ML model  
**Task:** next-day (t+1) 24-h mean PM2.5 (µg/m³), Option A  
**Best model:** no  
**Seed:** 42

## Test-set performance (point + 95% bootstrap CI, n_boot=1000)

| metric | value | 95% CI |
|---|---|---|
| RMSE | 30.643 | [25.490, 36.171] |
| MAE | 21.189 | [18.268, 24.396] |
| R² | 0.797 | [0.743, 0.843] |
| bias | 1.075 | [-2.744, 5.146] |
| n (test) | 219 | — |

## Features

45 day-t features (no t+1, no same-station o3/pm10, no satellite PM2.5). See reports/feature_availability.csv.

## Notes
- ERA5 = reanalysis used as a day-t predictor proxy.
- BLH for the 2024 H1 gap is sourced from native ERA5 (Copernicus CDS); `era5_blh_imputed_day` is now constant 0 (no imputed hours remain). Fallback when CDS is unavailable: train-years-only (<=2023) month×hour climatology.
