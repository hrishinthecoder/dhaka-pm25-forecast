# Phase 3 — Target Construction

**Definition (verbatim):** `pm25_next_day_daily_mean` = the daily-MEAN PM2.5 of sensor 24434 on day **t+1**, predicted from data available through day **t**. Daily mean is over the LOCAL calendar day (Asia/Dhaka, UTC+6); a day counts only with **≥18 valid hourly observations**. Not an AQI, not a same-hour +24h shift.

- Single-station source: `raw/openaq/sensor_24434_pm25.parquet` (sensor 24434, US diplomatic-post reference monitor, Dhaka (~2 km from centre)).

- Qualifying days (≥18 valid hrs): **1899**
- Consecutive (t, t+1) supervised pairs: **1746**
- Origin-day span: **2016-11-10 → 2025-03-23**
- Target mean/std: 90.7 / 68.6 µg/m³


## Supervised pairs per origin-day year

| Year | Pairs |
|---|---|
| 2016 | 44 |
| 2017 | 207 |
| 2018 | 133 |
| 2019 | 219 |
| 2020 | 207 |
| 2021 | 193 |
| 2022 | 200 |
| 2023 | 230 |
| 2024 | 232 |
| 2025 | 81 |
| **Total** | **1746** |

Saved table: `processed/supervised_daily.parquet` (shape (1746, 62)).