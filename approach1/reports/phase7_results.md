# Phase 7 — Explainability (SHAP on LightGBM)

_Generated 2026-06-10._  TreeExplainer on the dev set.

## Feature-group attribution (which signal carries the forecast)

| Group | Σ mean\|SHAP\| | share % |
|---|---|---|
| era5_meteorology | 39.222 | 58.0% |
| pm25_lags | 15.657 | 23.2% |
| pm25_rolling | 9.161 | 13.5% |
| calendar_cyclical | 3.417 | 5.1% |
| fire | 0.166 | 0.2% |
| outlier_flag | 0.000 | 0.0% |

_Figure: `figures/shap_group_attribution.png`._


## Top 20 individual features (mean\|SHAP\|)

| Rank | Feature | mean\|SHAP\| |
|---|---|---|
| 1 | `era5_dew_point_2m_d0` | 19.907 |
| 2 | `pm25_lag_1` | 15.385 |
| 3 | `pm25_roll7_mean` | 3.692 |
| 4 | `era5_dew_point_2m_roll3` | 3.473 |
| 5 | `day_of_year_cos` | 3.271 |
| 6 | `era5_cloud_cover_d0` | 2.528 |
| 7 | `era5_wind_gusts_10m_d0` | 2.469 |
| 8 | `era5_surface_pressure_roll3` | 2.227 |
| 9 | `pm25_roll3_mean` | 1.696 |
| 10 | `era5_precipitation_d0` | 1.640 |
| 11 | `era5_precipitation_roll3` | 1.353 |
| 12 | `era5_surface_pressure_d0` | 1.284 |
| 13 | `pm25_roll7_min` | 0.899 |
| 14 | `pm25_roll3_min` | 0.729 |
| 15 | `era5_cloud_cover_roll3` | 0.621 |
| 16 | `era5_wind_speed_10m_d0` | 0.612 |
| 17 | `pm25_roll7_max` | 0.602 |
| 18 | `era5_wind_direction_10m_roll3` | 0.540 |
| 19 | `pm25_roll14_mean` | 0.427 |
| 20 | `era5_boundary_layer_height_roll3` | 0.421 |

_Figures: `figures/shap_beeswarm.png`, `figures/shap_bar.png`._


## Interpretation

The forecast is dominated by **era5_meteorology** (58% of total attribution), followed by **pm25_lags** (23%). A lag-dominated model is persistence-like memory; meteorology dominance (boundary-layer height, wind, humidity) is the physically interpretable signal reviewers want; fire contribution flags biomass-burning episodes. See the group table above for the actual split on this run.