"""
phase0_target_24434.py — Standalone profile of the chosen SINGLE-STATION target.

WHAT THIS DOES
    Phase 0 found the pooled `openaq_pm25` was a multi-site spatial mean. The human
    decision (2026-06-10) is to take the target from ONE raw sensor — 24434, the
    US diplomatic-post reference monitor (~2 km from centre). This script profiles
    THAT sensor alone, exactly as the future target pipeline will see it:
        - valid hourly observations per year (2016–2026) and continuity,
        - local-day (Asia/Dhaka) coverage under the >=18-valid-hours rule,
        - the resulting consecutive (t, t+1) supervised-pair count.

WHY IT MATTERS
    Switching from the pooled column to one sensor reduces coverage. We must know the
    real single-station N before committing to Phase 1 — a forecasting study lives or
    dies on having enough honest supervised pairs.

Read-only: prints summaries, writes nothing.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SENSOR = BASE / "raw" / "openaq" / "sensor_24434_pm25.parquet"
LOCAL_TZ = "Asia/Dhaka"
MIN_HOURS = 18
PM_MIN, PM_MAX = 0.0, 1000.0  # config physical_bounds — hard sanity filter

s = pd.read_parquet(SENSOR, columns=["datetime_utc", "value", "location_id",
                                     "location_name", "sensor_id"])
print("file:", SENSOR.name)
print("raw rows:", len(s),
      " location_id:", int(s["location_id"].iloc[0]),
      " name:", s["location_name"].iloc[0])

# --- hourly series: floor to UTC hour, average any sub-hourly duplicates ------
t = pd.to_datetime(s["datetime_utc"], utc=True).dt.floor("h")
ser = pd.Series(s["value"].values, index=t).groupby(level=0).mean()
print("unique UTC hours (raw):", len(ser))

# --- physical-bounds sanity filter (config: 0 <= pm25 <= 1000) ----------------
before = len(ser)
ser = ser[(ser >= PM_MIN) & (ser <= PM_MAX)]
print(f"dropped out-of-bounds [{PM_MIN},{PM_MAX}]:", before - len(ser),
      " -> valid hours:", len(ser))

# --- valid hourly obs per year ------------------------------------------------
yr = ser.index.year
per_year = pd.Series(1, index=ser.index).groupby(yr).size()
print("\nVALID HOURLY OBS PER YEAR (UTC):")
for y in range(2016, 2027):
    print(f"  {y}: {int(per_year.get(y, 0))}")
print("  total:", int(len(ser)))
yrs_present = sorted(set(int(y) for y in yr))
print("years present:", yrs_present)
gaps = [y for y in range(min(yrs_present), max(yrs_present) + 1) if y not in yrs_present]
print("missing whole years between first and last:", gaps if gaps else "none (continuous)")

# --- local-day coverage + supervised (t, t+1) pairs ---------------------------
local = ser.index.tz_convert(LOCAL_TZ)
day = pd.Series(local.date, index=ser.index)
hours_per_day = day.groupby(day).size()  # valid hours per local day
good = hours_per_day[hours_per_day >= MIN_HOURS].index
good = pd.to_datetime(sorted(good))
print(f"\nlocal days with >=1 valid hour: {hours_per_day.shape[0]}")
print(f"local days with >={MIN_HOURS} valid hours: {len(good)}")
if len(good):
    print("first qualifying day:", good.min().date(), " last:", good.max().date())
gd = pd.Series(good)
pairs = int((gd.shift(-1) == gd + pd.Timedelta(days=1)).sum())
print("consecutive (t, t+1) supervised pairs:", pairs)

# qualifying-day count per year (the rows that actually feed the model)
gy = pd.Series(good).dt.year.value_counts().sort_index()
print("\nQUALIFYING LOCAL DAYS (>=18h) PER YEAR:")
for y in range(2016, 2027):
    print(f"  {y}: {int(gy.get(y, 0))}")
