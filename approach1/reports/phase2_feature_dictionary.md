# Phase 2 — Feature Dictionary

_Generated 2026-06-10._  Target = next-day daily-mean PM2.5 of sensor 24434 (single station).

**Leakage rule:** every feature uses information available **through origin day t** to predict day **t+1**. The only forward-looking inputs are CALENDAR features of the target day t+1 — deterministic and known in advance, so they leak nothing.


Total engineered features: **59**. Core-model features (groups ['era5_meteorology', 'calendar_cyclical', 'pm25_lags', 'pm25_rolling', 'fire']): **51**.


| Feature | Group | In core? | Temporal availability |
|---|---|---|---|
| `era5_temperature_2m_d0` | era5_meteorology | yes | day t (most recent observed) |
| `era5_temperature_2m_roll3` | era5_meteorology | yes | trailing window ending day t |
| `era5_relative_humidity_2m_d0` | era5_meteorology | yes | day t (most recent observed) |
| `era5_relative_humidity_2m_roll3` | era5_meteorology | yes | trailing window ending day t |
| `era5_dew_point_2m_d0` | era5_meteorology | yes | day t (most recent observed) |
| `era5_dew_point_2m_roll3` | era5_meteorology | yes | trailing window ending day t |
| `era5_wind_speed_10m_d0` | era5_meteorology | yes | day t (most recent observed) |
| `era5_wind_speed_10m_roll3` | era5_meteorology | yes | trailing window ending day t |
| `era5_wind_gusts_10m_d0` | era5_meteorology | yes | day t (most recent observed) |
| `era5_wind_gusts_10m_roll3` | era5_meteorology | yes | trailing window ending day t |
| `era5_wind_direction_10m_d0` | era5_meteorology | yes | day t (most recent observed) |
| `era5_wind_direction_10m_roll3` | era5_meteorology | yes | trailing window ending day t |
| `era5_surface_pressure_d0` | era5_meteorology | yes | day t (most recent observed) |
| `era5_surface_pressure_roll3` | era5_meteorology | yes | trailing window ending day t |
| `era5_precipitation_d0` | era5_meteorology | yes | day t (most recent observed) |
| `era5_precipitation_roll3` | era5_meteorology | yes | trailing window ending day t |
| `era5_cloud_cover_d0` | era5_meteorology | yes | day t (most recent observed) |
| `era5_cloud_cover_roll3` | era5_meteorology | yes | trailing window ending day t |
| `era5_boundary_layer_height_d0` | era5_meteorology | yes | day t (most recent observed) |
| `era5_boundary_layer_height_roll3` | era5_meteorology | yes | trailing window ending day t |
| `era5_shortwave_radiation_d0` | era5_meteorology | yes | day t (most recent observed) |
| `era5_shortwave_radiation_roll3` | era5_meteorology | yes | trailing window ending day t |
| `era5_vapour_pressure_deficit_d0` | era5_meteorology | yes | day t (most recent observed) |
| `era5_vapour_pressure_deficit_roll3` | era5_meteorology | yes | trailing window ending day t |
| `era5_boundary_layer_height_was_missing` | era5_meteorology | yes | day t (most recent observed) |
| `firms_count` | fire | yes | day t (most recent observed) |
| `firms_frp_sum` | fire | yes | day t (most recent observed) |
| `acag_pm25` | pm_adjacent_model_products | no | day t (most recent observed) |
| `omaq_carbon_monoxide` | cams_chemistry | no | day t (most recent observed) |
| `omaq_nitrogen_dioxide` | cams_chemistry | no | day t (most recent observed) |
| `omaq_sulphur_dioxide` | cams_chemistry | no | day t (most recent observed) |
| `omaq_ozone` | cams_chemistry | no | day t (most recent observed) |
| `omaq_dust` | pm_adjacent_model_products | no | day t (most recent observed) |
| `omaq_aerosol_optical_depth` | pm_adjacent_model_products | no | day t (most recent observed) |
| `omaq_pm2_5` | cams_pm25_benchmark | no | day t (most recent observed) |
| `pm25_lag_1` | pm25_lags | yes | day t and earlier |
| `pm25_lag_2` | pm25_lags | yes | day t and earlier |
| `pm25_lag_3` | pm25_lags | yes | day t and earlier |
| `pm25_lag_7` | pm25_lags | yes | day t and earlier |
| `pm25_roll3_mean` | pm25_rolling | yes | trailing window ending day t |
| `pm25_roll3_std` | pm25_rolling | yes | trailing window ending day t |
| `pm25_roll3_min` | pm25_rolling | yes | trailing window ending day t |
| `pm25_roll3_max` | pm25_rolling | yes | trailing window ending day t |
| `pm25_roll7_mean` | pm25_rolling | yes | trailing window ending day t |
| `pm25_roll7_std` | pm25_rolling | yes | trailing window ending day t |
| `pm25_roll7_min` | pm25_rolling | yes | trailing window ending day t |
| `pm25_roll7_max` | pm25_rolling | yes | trailing window ending day t |
| `pm25_roll14_mean` | pm25_rolling | yes | trailing window ending day t |
| `pm25_roll14_std` | pm25_rolling | yes | trailing window ending day t |
| `pm25_roll14_min` | pm25_rolling | yes | trailing window ending day t |
| `pm25_roll14_max` | pm25_rolling | yes | trailing window ending day t |
| `month_sin` | calendar_cyclical | yes | target day t+1 (deterministic) |
| `month_cos` | calendar_cyclical | yes | target day t+1 (deterministic) |
| `day_of_week_sin` | calendar_cyclical | yes | target day t+1 (deterministic) |
| `day_of_week_cos` | calendar_cyclical | yes | target day t+1 (deterministic) |
| `day_of_year_sin` | calendar_cyclical | yes | target day t+1 (deterministic) |
| `day_of_year_cos` | calendar_cyclical | yes | target day t+1 (deterministic) |
| `is_weekend` | calendar_cyclical | yes | target day t+1 (deterministic) |
| `is_public_holiday` | calendar_cyclical | yes | target day t+1 (deterministic) |

**Justification:** no feature reads any value dated after day t except deterministic calendar fields of t+1; PM2.5 rolling windows end at day t (target day excluded); FIRMS `firms_frp_sum` is 0-filled where no fire was detected (config `preprocessing.firms_frp_sum_fill`).