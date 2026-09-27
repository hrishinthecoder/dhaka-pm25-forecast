# Ingest Validation — Dhaka PM2.5 Master Rebuild

Notebook: `notebooks/01_data_acquisition.ipynb` · Source of truth: `config/sources.yaml`

## What the notebook builds

A continuous **hourly UTC** modeling master over `date_range` (2016-01-01 →
2026-01-01, end exclusive), saved to:

- `data/processed/dhaka_aq_master_rebuilt.parquet`
- `data/processed/dhaka_aq_master_rebuilt.csv`

| Block | Source | Columns | Notes |
|-------|--------|---------|-------|
| Ground truth + covariates | OpenAQ v3 `/sensors/{id}/hours` | `openaq_pm25` (**TARGET**), `openaq_pm10`, `openaq_o3`, `openaq_co`, `openaq_no2`, `openaq_so2` | µg/m³ for PM; covariates only. **No CO2.** |
| Meteorology | Open-Meteo ERA5 archive | `era5_*` (10 hourly vars) | **ERA5 reanalysis** — NOT station observations |
| Fire activity | NASA FIRMS area API | `firms_count`, `firms_frp_sum` | daily, broadcast to every hour of the day |
| Calendar | pure (Step 4) | `hour`, `day_of_week`, `month`, `is_weekend`, `is_rush_hour`, `hour_sin`, `hour_cos` | weekend = Fri/Sat (Bangladesh) |
| Calendar (Islamic/national) | `holidays` lib + embedded Ramadan table | `is_holiday`, `is_ramadan` | Asia/Dhaka local date |

Time stored as `datetime_utc` (+ `datetime_local` = Asia/Dhaka). `omaq_ammonia`
and any fully-empty columns are dropped before save.

## Reference & gates

Validated against `data/reference/dhaka_aq_master.csv` (248,953 rows; continuous
PM2.5 2016–2025) via `tests/test_against_reference.py` (thresholds unchanged):

- Ground-truth PM2.5 column auto-detected: `openaq_pm25` (matches `^openaq_pm2?_?5$`).
- **Yearly coverage** ≥ 70% of reference valid PM2.5 hours per year (2016–2025).
- **Median PM2.5 drift** ≤ 15% vs reference.
- **No forbidden columns** (`co2`, `ammonia`).

Run: `python -m pytest tests/test_against_reference.py -v` (Step 6 cell runs this).

## Design decisions (live-API faithfulness)

- **OpenAQ v3 auth** = `X-API-Key` header. Hourly via `/v3/sensors/{id}/hours`;
  timestamp = `period.datetimeFrom.utc`; paginated on `meta.found` (limit 1000);
  chunked by **calendar month** with window params **`datetime_from`/`datetime_to`**
  (the v3 hours endpoint ignores `date_from`/`date_to`); raw pages cached to
  `data/raw/openaq/`. A sensor that fails 3 windows in a row is abandoned.
  Covariates are averaged across stations per hour in the merge; the **PM2.5 target is
  single-sensor 24434 only**, never the pooled cross-station mean (see below).
- **Stations** = fixed-site monitors `[22, 2445, 8415, 3194367]` for *covariates*; ~18
  scattered low-cost Smart Air units excluded (uncalibrated, start ~2026). The **PM2.5
  target is sensor 24434 (loc 8415) alone** — the 2016-11→2025-03 single-station spine
  (US Diplomatic Post, ~2 km from centre). The pooled mean is **not** used as the target;
  the single-station target is swapped in by `fusion/swap_target.py`.
- **ERA5** chunked by year, cached to `data/raw/open_meteo/`, `timezone=UTC`.
- **FIRMS** paged by ≤**5-day** windows (area CSV API caps at 5); windows older
  than ~60 days use the `_SP` archive product, recent windows use NRT; cached to
  `data/raw/firms/`. HTTP 400 / missing `FIRMS_MAP_KEY` → fire columns = 0 (non-fatal).
- Retry/backoff (exponential, honours `Retry-After`) on all HTTP fetchers; 408
  fast-fails (retrying the same heavy query just times out again).

## Execution status — VERIFIED (2026-06-11)

Ran end-to-end (Restart kernel & run all). Output **87,672 rows × 27 cols**
(26 data + `era5_blh_imputed` flag), continuous hourly **2016-01-01 → 2025-12-31
UTC**; **70,288** valid PM2.5 hours in the pooled rebuild (the `test_against_reference`
ingest gate runs on this pooled master). All 4 gates **PASS**.

> Note: the **single-station modeling target** (sensor 24434, `*_singlestation.parquet`)
> has **47,656** valid PM2.5 hours, ending 2025-03-24 — see `reports/data_notes.md`.

### BLH H1-2024 gap — imputed
`era5_boundary_layer_height` is missing for **2024-01-01 → 2024-06-30** (4,368 h).
This is a genuine **ERA5/Open-Meteo data gap**, not a fetch error: a fresh re-pull
returns null for that window across all models (era5 / era5_seamless / best_match),
and the **reference master has the identical 4,368-null hole**. Because 2024 is the
validation year and BLH drives PM2.5 accumulation, the hole is filled by
**month×hour climatology** (mean BLH per calendar-month/hour over all years with
data; imputed mean 496.9 m vs observed 464.6 m). A boolean column
**`era5_blh_imputed`** flags the 4,368 imputed rows for transparency. The
imputation is baked into `merge.py` (`_impute_blh`, called inside `build_master`),
so a notebook "Restart & run all" reproduces it automatically.

| Metric | Reference | Rebuilt | Pass? |
|--------|-----------|---------|-------|
| Median PM2.5 (µg/m³) | 60.0 | 64.0 | ✅ 6.7% drift (≤15%) |
| Coverage 2016–2025 | — | 90–139% of ref/yr | ✅ all ≥70% |
| Forbidden columns | none | none | ✅ none |

Per-year valid PM2.5 hours (rebuilt vs reference): 2016 6051/6706, 2017 6579/5605,
2018 4993/4993, 2019 8047/5828, 2020 7918/5759, 2021 7267/5333, 2022 7595/5466,
2023 7340/5887, 2024 8373/6935, 2025 6125/6275.
