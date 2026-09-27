# Model card — xgboost

**Type:** ML model  
**Task:** next-day (t+1) 24-h mean PM2.5 (µg/m³), Option A  
**Best model:** YES  
**Seed:** 42

## Test-set performance (point + 95% bootstrap CI, n_boot=1000)

| metric | value | 95% CI |
|---|---|---|
| RMSE | 30.804 | [25.925, 36.140] |
| MAE | 21.416 | [18.566, 24.485] |
| R² | 0.795 | [0.732, 0.841] |
| bias | 1.080 | [-2.613, 5.148] |
| n (test) | 219 | — |

## Tuning (TimeSeriesSplit n=5 on train+val; refit on train)

```json
{
  "best_params": {
    "learning_rate": 0.03,
    "max_depth": 3,
    "n_estimators": 300
  },
  "cv_rmse": 31.142869810717826
}
```

## AQI classification (US EPA 2024)

- accuracy: 0.658
- macro-F1: 0.503

## Features

45 day-t features (no t+1, no same-station o3/pm10, no satellite PM2.5). See reports/feature_availability.csv.

## Notes
- ERA5 = reanalysis used as a day-t predictor proxy.
- BLH for the 2024 H1 gap is sourced from native ERA5 (Copernicus CDS); `era5_blh_imputed_day` is now constant 0 (no imputed hours remain). Fallback when CDS is unavailable: train-years-only (<=2023) month×hour climatology.
