"""
phase0_audit.py — Phase 0 data verification for the Dhaka PM2.5 forecasting project.

WHAT THIS DOES
    Re-derives, directly from the data files, every fact the project's CLAUDE.md and
    master prompt claim about `dhaka_aq_master.parquet`. It prints compact summaries
    only (never dumps raw rows) so the numbers can be transcribed into
    reports/phase0_data_audit.md and checked against the stated "ground truth".

WHY IT MATTERS
    Phase 0 is the gate. If any number here contradicts the documented facts —
    especially the single-station claim — the build must STOP. Verifying instead of
    assuming is the whole point of this phase.

This script is read-only: it opens data files, computes statistics, prints them.
It writes nothing and modifies nothing.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent  # project root (parent of reports/)
MASTER = BASE / "processed" / "dhaka_aq_master.parquet"
OPENAQ_DIR = BASE / "raw" / "openaq"
MANIFEST = BASE / "manifest.json"
LOCAL_TZ = "Asia/Dhaka"          # UTC+6, no DST — "next day" means a citizen's local day
GROUND_TRUTH = "openaq_pm25"     # the ONLY target
MIN_HOURS = 18                   # a local day counts only with >=18 valid hourly obs


def hr(title):
    """Print a section header so the captured output is easy to read."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------
# 1. STRUCTURE: shape, columns, dtypes, forbidden columns
# --------------------------------------------------------------------------
df = pd.read_parquet(MASTER)
hr("1. STRUCTURE")
print("shape:", df.shape)
print("n_cols:", len(df.columns))
print("columns:", list(df.columns))
print("\ndtypes:")
print(df.dtypes.to_string())

cols = list(df.columns)
co2 = [c for c in cols if "co2" in c.lower() or "carbon_dioxide" in c.lower()]
station = [c for c in cols if any(k in c.lower() for k in
                                  ["station", "location", "sensor", "site"])]
latlon = [c for c in cols if any(k in c.lower() for k in ["latitude", "longitude"])
          and "direction" not in c.lower()]
print("\nCO2-like columns (expect NONE):", co2)
print("station/location/sensor columns (expect NONE):", station)
print("lat/lon columns (expect NONE):", latlon)
print("carbon_monoxide columns (expected covariate):",
      [c for c in cols if "carbon_monoxide" in c.lower()])

# --------------------------------------------------------------------------
# 2. TIME INDEX: range + granularity
# --------------------------------------------------------------------------
hr("2. TIME INDEX")
t = pd.to_datetime(df["datetime_utc"], utc=True)
print("datetime_utc dtype:", df["datetime_utc"].dtype)
print("min:", t.min(), " max:", t.max())
print("n_rows:", len(t), " n_unique_timestamps:", t.nunique())
deltas = t.sort_values().diff().dropna().value_counts().head(5)
print("most common consecutive deltas (expect 1h):")
print(deltas.to_string())
has_local = "datetime_local" in cols
print("datetime_local present:", has_local)

# --------------------------------------------------------------------------
# 3. GROUND TRUTH availability per year
# --------------------------------------------------------------------------
hr("3. openaq_pm25 VALID OBS PER YEAR")
gt = df[[ "datetime_utc", GROUND_TRUTH]].copy()
gt["year"] = t.dt.year.values
per_year = gt.dropna(subset=[GROUND_TRUTH]).groupby("year").size()
print(per_year.to_string())
print("total valid openaq_pm25:", int(df[GROUND_TRUTH].notna().sum()))

# --------------------------------------------------------------------------
# 4. HOURLY CORRELATIONS vs openaq_pm25
# --------------------------------------------------------------------------
hr("4. HOURLY PEARSON r vs openaq_pm25")
for c in ["omaq_pm2_5", "acag_pm25", "omaq_pm10", "openaq_pm10"]:
    if c in cols:
        sub = df[[GROUND_TRUTH, c]].dropna()
        r = sub[GROUND_TRUTH].corr(sub[c]) if len(sub) > 2 else float("nan")
        print(f"{c:14s} r={r:.4f}  n_overlap={len(sub)}")
    else:
        print(f"{c:14s} NOT PRESENT")

# --------------------------------------------------------------------------
# 5. MISSINGNESS per column
# --------------------------------------------------------------------------
hr("5. MISSINGNESS % PER COLUMN (sorted)")
miss = (df.isna().mean() * 100).sort_values(ascending=False)
print(miss.round(2).to_string())
print("\ncolumns >95% missing (flag for drop):",
      list(miss[miss > 95].index))

# --------------------------------------------------------------------------
# 6. DAILY COVERAGE on local calendar day + consecutive (t, t+1) pairs
# --------------------------------------------------------------------------
hr("6. DAILY COVERAGE (Asia/Dhaka) + SUPERVISED PAIRS")
local = t.dt.tz_convert(LOCAL_TZ)
day = local.dt.date
valid = df[GROUND_TRUTH].notna().values
cov = pd.DataFrame({"day": day.values, "valid": valid})
hours_per_day = cov.groupby("day")["valid"].sum()
good_days = hours_per_day[hours_per_day >= MIN_HOURS].index
good_days = pd.to_datetime(sorted(good_days))
print(f"days with >={MIN_HOURS} valid hours:", len(good_days))
print("total distinct local days with >=1 valid hour:", int((hours_per_day >= 1).sum()))
# consecutive (t, t+1) pairs: both day t and day t+1 clear the threshold
gd = pd.Series(good_days)
next_day_present = gd.shift(-1) == gd + pd.Timedelta(days=1)
n_pairs = int(next_day_present.sum())
print("consecutive (t, t+1) supervised pairs:", n_pairs)
if len(good_days):
    print("first good day:", good_days.min().date(), " last:", good_days.max().date())

# --------------------------------------------------------------------------
# 7. SENSOR FILE INTROSPECTION (for single-station step)
# --------------------------------------------------------------------------
hr("7. OPENAQ pm25 SENSOR FILES — schema sample")
pm25_files = sorted(OPENAQ_DIR.glob("sensor_*_pm25.parquet"))
print("n pm25 sensor files:", len(pm25_files))
sample = pd.read_parquet(pm25_files[0])
print("sample file:", pm25_files[0].name)
print("sample columns:", list(sample.columns))
print("sample dtypes:")
print(sample.dtypes.to_string())
print(sample.head(3).to_string())
