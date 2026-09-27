# Phase 0 — Data Verification Audit

**Project:** Dhaka next-day PM2.5 forecasting
**Date:** 2026-06-10
**Status:** ⛔ **STOP — blocking discrepancy found (single-station assumption is FALSE).** Awaiting human review before Phase 1.
**Inputs verified:** `processed/dhaka_aq_master.parquet`, `raw/openaq/*`, `manifest.json`, `data_dictionary.md`, `config/config.yaml`, `requirements.txt`
**Reproduce:** `python reports/phase0_audit.py` and `python reports/phase0_station.py`

---

## 0. Headline verdict

| Constraint (CLAUDE.md §2 / master prompt) | Result | Verdict |
|---|---|---|
| **Single station** | `openaq_pm25` = **hourly mean of up to 20 sensors across 14+ distinct locations (0.6–17.3 km apart)** | ❌ **VIOLATED → STOP** |
| Target is PM2.5 only; no CO₂ | No CO₂ column; only `omaq_carbon_monoxide` (covariate) | ✅ |
| Shape 248,953 × 39 | 248,953 × 39 | ✅ |
| Hourly granularity | 248,952 / 248,952 consecutive deltas = 1h | ✅ |
| Ground truth continuous 2016–2026, no multi-year gap | 4,993–6,935 valid hrs/yr, every year | ✅ |
| Supervised (t,t+1) pairs ~2,300–2,550 | **2,306** | ✅ (in band; below the ~2,355 stated — see §6) |
| All concentration features µg/m³ | Confirmed, no cross-source mismatch | ✅ |
| `omaq_ammonia` 100% empty → drop | 100.00% missing | ✅ |

**The single-station violation is a hard STOP per master-prompt constraint #2: "If multiple sensors at different sites were pooled into one series, STOP and report — it breaks the single-point framing."**

---

## 1. Structure

- **Shape:** 248,953 rows × 39 columns. ✅ matches stated facts.
- **No CO₂ column** (searched `co2`, `carbon_dioxide`): none. Only carbon column is `omaq_carbon_monoxide` (CO, covariate). ✅
- **No station / location / sensor / lat / lon column** in the master: none. ✅ (identifiers were dropped in the merge, as documented — which is *why* single-station had to be reverse-engineered from `raw/openaq/`.)
- All non-time columns are `float64`; `datetime_utc` is `datetime64[us, UTC]`, `datetime_local` is `datetime64[us, Asia/Dhaka]`. ✅

## 2. Time index

- **Range:** 1998-01-01 00:00 UTC → 2026-05-27 00:00 UTC. (Meteorology back to 1998; ground truth 2016+ — consistent with CLAUDE.md.)
- **Granularity:** hourly — 248,952 of 248,952 consecutive gaps are exactly 1h; all 248,953 timestamps unique. ✅
- `datetime_local` (Asia/Dhaka) present and used for daily aggregation. ✅

## 3. Ground-truth (`openaq_pm25`) valid observations per year

| Year | Valid hours | | Year | Valid hours |
|---|---|---|---|---|
| 2016 | 6,706 | | 2022 | 5,466 |
| 2017 | 5,605 | | 2023 | 5,887 |
| 2018 | 4,993 | | 2024 | 6,935 |
| 2019 | 5,828 | | 2025 | 6,275 |
| 2020 | 5,759 | | 2026 | 3,496 (partial yr) |
| 2021 | 5,333 | | **Total** | **62,283** |

Continuous 2016→2026, **no multi-year gap**. ✅ (2018 is the thinnest full year at 4,993 hrs.)

## 4. Hourly Pearson r vs `openaq_pm25`

| Feature | r | overlap n | Note |
|---|---|---|---|
| `omaq_pm2_5` (CAMS model PM2.5) | **0.754** | 23,620 | Near-circular → exclude from core; ablation only. ✅ matches stated ~0.754 |
| `acag_pm25` (annual satellite grid) | **0.149** | 62,283 | Coarse anchor only. ✅ matches stated ~0.149 |
| `omaq_pm10` (CAMS model PM10) | 0.690 | 23,620 | Ablation group |
| `openaq_pm10` (measured) | 0.984 | 1,552 | Co-measured PM, sparse → excluded (leakage-adjacent) |

## 5. Missingness per column

**Flagged >95% missing:** `omaq_ammonia` (100.00%), `openaq_pm10` (99.38%), **`firms_frp_sum` (98.69%)**, `openaq_o3` (96.63%).

Other notable: all `omaq_*` chemistry = 86.58% missing; `openaq_pm25` (target) = 74.98% missing (i.e. 25% of hours have ground truth); `power_*` ≈ 10.6–12.0%; `era5_boundary_layer_height` = 1.75%; remaining `era5_*`, `firms_count`, `acag_pm25` = 0.00%.

⚠️ **Discrepancy to resolve:** `firms_frp_sum` is **98.69% missing** yet `config.yaml` lists it in the **core** `fire` feature group (`features.fire`). `firms_count` is fully populated (0% — zero-filled when no fire). Recommend: keep `firms_count` as core; treat `firms_frp_sum` as near-empty (impute 0 / treat as event flag), or drop. Needs a decision before Phase 2.

## 6. Daily coverage (Asia/Dhaka local day) + supervised pairs

- Days with **≥18 valid hourly obs:** **2,489** (CLAUDE.md stated 2,537 — **48 fewer**; minor, likely a local-day boundary / inclusivity difference. Not blocking, but noted.)
- Distinct local days with ≥1 valid hour: 2,870.
- **Consecutive (t, t+1) supervised pairs: 2,306** — within the master-prompt acceptance band (2,300–2,550), slightly below the ~2,355 stated in CLAUDE.md.
- First qualifying day: **2016-03-11**; last: **2026-05-26**.

## 7. ⛔ Single-station verification (CRITICAL)

`raw/openaq/` contains **20 distinct pm25 sensor files** (+ 1 pm10, 1 o3). `_sensors.csv` and the manifest show these sit at **20 distinct `location_id`s, 0.6–17.3 km from the city centre** — e.g. US Diplomatic Post, SPARTAN Dhaka University (9 km), Jahangirnagar University (17 km), RAJUK Uttara (7.6 km), and many Smart Air Bangladesh neighbourhood units.

**How `openaq_pm25` was built (reverse-engineered, since identifiers were dropped):**

| Hypothesis | Match to master `openaq_pm25` |
|---|---|
| (A) single dominant sensor 24434 alone | 46,630 / 62,283 = **74.87%** |
| **(B) hourly MEAN across all sensors present** | **62,283 / 62,283 = 100.00%** ✅ |

→ **`openaq_pm25` is the hourly arithmetic mean of every OpenAQ pm25 sensor reporting that hour.**

Blending magnitude:
- **0** master hours are unexplained by hypothesis B (perfect reconstruction).
- 57,512 hrs (92.3%) — exactly one sensor reported → master = that sensor.
- **4,771 hrs (7.7%) — ≥2 sensors averaged**, up to **12 sensors blended in a single hour**.
- 46,629 hrs (**74.9%**) — sourced solely from dominant sensor **24434** (US-embassy-area "Dhaka", ~2 km).
- Remaining ~25.1% — other sites, singly or blended.
- **14 distinct `location_id`s** contribute matched hours: 2445, 8415, 3194367, 5105944, 6157905, 6234078, 6234363, 6236590, 6240023, 6240773, 6242079, 6242232, 6251395, 6271076.

**Conclusion:** This is **NOT a single station**. It is a spatially-pooled, geographically-spread multi-sensor mean across ~14 sites spanning ~17 km, with a time-varying, sensor-mix-dependent number of contributors. The "single-point temporal forecasting" framing **does not hold as-is**. **STOP and report** (this document).

## 8. Provenance & units audit

- **Manifest location:** found at **project root `./manifest.json`**, NOT at `raw/manifest.json` (as `config.yaml` and the master prompt state). Its `file` paths also point to a `…\Research (Warmup)_Team\data\raw\openaq\…` layout (a `data/` parent that does not exist here; actual files are at `raw/…`). Paths are stale relative to this copy — **note for Phase 1 path config.** Non-blocking.
- **Sources merged into master (per manifest):** openaq, open_meteo_weather (ERA5), open_meteo_air_quality (CAMS), nasa_power, nasa_firms, acag. `cams_eac4`, `gee`, `merra2` are **absent from the manifest** → confirmed collected-but-not-merged / **out of scope**. ✅
- **Units:** all concentration columns are µg/m³ (`openaq_*`, `omaq_*`, `acag_pm25`), AOD unitless, met in physical SI-ish units per `data_dictionary.md`. **No cross-source concentration unit mismatch.** ✅
- **Coarser-than-hourly cadence (broadcast → artificial autocorrelation, anchor-only):** `acag_pm25` (annual/monthly satellite grid; manifest entries carry no temporal cadence, only url/file → confirms static-ish broadcast; r=0.149). Treat as coarse anchor only. ✅
- **FIRMS aggregation check:** `firms_count` / `firms_frp_sum` derived from NASA FIRMS area CSV API, bbox **`90.15,23.55,90.65,24.05`** (~0.5°×0.5°, ≈55 km box around Dhaka), 5 satellite streams (MODIS Terra/Aqua, VIIRS SNPP/NOAA-20). The **exact temporal aggregation window** (hourly vs daily count) is **not explicitly recorded** in the manifest entry — recommend confirming before relying on `firms_count` timing in Phase 2. `firms_count` 0% missing (0 when no detection); `firms_frp_sum` 98.69% missing (populated only on detection).

## 9. Environment

- Python **3.12.10** (host); dependencies installed from provided `requirements.txt` (exit 0). Resolved key versions: pandas 3.0.3, numpy, scipy, scikit-learn, lightgbm, xgboost, optuna, shap, holidays, matplotlib, seaborn, pyyaml, joblib, pyarrow, streamlit, torch.
- `config/config.yaml` and `requirements.txt` read, **not modified**. Master files **not moved or overwritten**.
- **Pending setup (deferred until the §7 decision):** git init; `requirements.lock` via `pip freeze`; `src/`, `app/`, `figures/`, `artifacts/`, `.claude_memory/` scaffolding (all Phase 1).
- Note: the context-mode sandbox runs Python 3.14 without pyarrow, so all parquet analysis ran on the host interpreter.

---

## 10. Required decisions before Phase 1 (STOP gate)

1. **Single-station framing (BLOCKING).** `openaq_pm25` is a multi-site spatial mean, not one station. Options:
   - **(a) Reconstruct a single-sensor target** from `raw/openaq/sensor_24434_pm25.parquet` (US-embassy-area, 74.9% of current coverage, the most continuous) and rebuild the daily target/pairs from it alone — preserves the single-point framing honestly, at some loss of coverage.
   - **(b) Keep the pooled mean** but **re-frame the manuscript** as a *city-representative / multi-station-averaged* PM2.5 forecast (drop all single-station language; describe the spatial pooling and its 0.6–17.3 km footprint explicitly).
   - **(c) Other** (e.g. restrict to a subset of nearby sensors). 

   This changes the target definition, so it must be settled before Phase 1/3.
2. **`firms_frp_sum`** (98.69% missing) is in the core feature group — confirm keep-as-flag / impute-0 / drop.
3. Confirm tolerance on the day-count (2,489 vs stated 2,537) and pair-count (2,306 vs ~2,355) — within band, just flagging.

**No Phase 1 work has begun. Awaiting your decision on item 1.**

---

# ADDENDUM — Phase-0 decision applied (2026-06-10)

**Decision (human):** adopt a **SINGLE-STATION target from sensor 24434 only** (US diplomatic-post reference monitor, loc 8415, ~2 km from centre). The pooled `openaq_pm25` is **not** the target — retained only as a city-mean reference.

**Config patch applied** to `config/config.yaml`: `target.source=raw_sensor`, `target.sensor_file=raw/openaq/sensor_24434_pm25.parquet`, `target.sensor_id=24434`, `target.station_label`; `preprocessing.firms_frp_sum_fill=0` (keep `firms_frp_sum`, impute 0 where `firms_count==0`); `paths.manifest` corrected to `manifest.json`. **Reproduce:** `python reports/phase0_target_24434.py`.

## A1. Sensor 24434 standalone profile

- Raw rows / unique UTC hours: **52,217**. Of these **4,561 are NaN** (missing values inside the file) → **47,656 valid hourly observations**.
- Value sanity: **0 negatives, 0 values >1000, 0 out-of-bounds**; min 0, max 985 µg/m³, mean **91.7**, median 62, std 86 — physically plausible for Dhaka. (12 exact-zero hours — minor, revisit in Phase 2 stuck-sensor rule.)

### A2. Valid hourly obs per year (continuity)

| Year | Valid hrs | | Year | Valid hrs |
|---|---|---|---|---|
| 2016 | 1,188 (from Nov) | | 2021 | 5,333 |
| 2017 | 5,605 | | 2022 | 5,466 |
| 2018 | 4,993 | | 2023 | 5,887 |
| 2019 | 5,828 | | 2024 | 5,631 |
| 2020 | 5,759 | | 2025 | 1,966 (to Mar) |
| | | | 2026 | **0** |

**Continuous 2016–2025, no missing whole year.** But the span is **shorter than the pooled column**: 2016 starts late (Nov) and **the sensor stops at 2025-03-24** — no 2026 data, sparse 2025.

### A3. Local-day (Asia/Dhaka) coverage + supervised pairs (≥18h rule)

| Metric | Pooled (rejected) | **Sensor 24434 (chosen)** |
|---|---|---|
| Local days with ≥1 valid hr | 2,870 | 2,220 |
| Local days with ≥18 valid hrs | 2,489 | **1,899** |
| Consecutive (t, t+1) pairs | 2,306 | **1,746** |
| First / last qualifying day | 2016-03-11 / 2026-05-26 | **2016-11-10 / 2025-03-24** |

Qualifying days/yr (≥18h): 2016:47, 2017:227, 2018:166, 2019:236, 2020:228, 2021:216, 2022:217, 2023:241, 2024:239, 2025:82.

## A4. ⚠️ Two consequences of the single-station switch (need your eye before Phase 1)

1. **Supervised pairs drop to 1,746** — **below the master-prompt acceptance band (2,300–2,550)** and ~24% below the pooled 2,306. The "small-N problem" is now smaller-N. Still workable for tree models + walk-forward CV, but the manuscript must state N≈1,746 honestly and temper deep-model expectations further. **Confirm this reduced N is acceptable.**
2. **Coverage ends 2025-03-24** (sensor went offline). The configured `final_holdout_months: 12` would now be roughly **2024-03 → 2025-03**, and 2025 contributes only 82 qualifying days — a thinner, older holdout than the pooled series implied. **Confirm the holdout window**, or consider defining it on the last 12 months of *available* data rather than wall-clock-recent.

**STOP again per your instruction. No Phase 1 work started. Awaiting confirmation on A4 (reduced N + holdout window).**
