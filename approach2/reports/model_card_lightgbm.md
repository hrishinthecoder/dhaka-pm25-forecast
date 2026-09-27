# Model card — lightgbm

**Type:** ML model  
**Task:** next-day (t+1) 24-h mean PM2.5 (µg/m³), Option A  
**Best model:** no  
**Seed:** 42

## Test-set performance (point + 95% bootstrap CI, n_boot=1000)

| metric | value | 95% CI |
|---|---|---|
| RMSE | 31.522 | [26.427, 37.072] |
| MAE | 21.794 | [18.970, 24.950] |
| R² | 0.785 | [0.723, 0.836] |
| bias | -0.264 | [-4.226, 3.867] |
| n (test) | 219 | — |

## Tuning (TimeSeriesSplit n=5 on train+val; refit on train)

```json
{
  "best_params": {
    "learning_rate": 0.03,
    "n_estimators": 300,
    "num_leaves": 31
  },
  "cv_rmse": 32.1945434349066
}
```

## Features

45 day-t features (no t+1, no same-station o3/pm10, no satellite PM2.5). See reports/feature_availability.csv.

## Notes
- ERA5 = reanalysis used as a day-t predictor proxy.
- BLH for the 2024 H1 gap is sourced from native ERA5 (Copernicus CDS); `era5_blh_imputed_day` is now constant 0 (no imputed hours remain). Fallback when CDS is unavailable: train-years-only (<=2023) month×hour climatology.
