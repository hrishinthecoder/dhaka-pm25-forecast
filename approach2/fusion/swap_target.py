"""Step 2 — swap the pooled openaq_pm25 target for single-sensor 24434.

Replaces ONLY the hourly `openaq_pm25` column of the master with sensor 24434's
hourly PM2.5, aligned on the master's UTC hourly index. Hours where 24434 has no
reading become NaN (no imputation, no fill from other stations). Every other
column is copied untouched. Writes a NEW file; the pooled original is preserved.

Run: ./.venv/Scripts/python.exe fusion/swap_target.py
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

SENSOR = Path(r"../Approach one/Machine Learning Model/raw/openaq/sensor_24434_pm25.parquet")
MASTER_IN = Path("data/processed/dhaka_aq_master_rebuilt.parquet")
OUT_PARQUET = Path("data/processed/dhaka_aq_master_singlestation.parquet")
OUT_CSV = Path("data/processed/dhaka_aq_master_singlestation.csv")

# --- load master, key on the UTC hourly index ---
m = pd.read_parquet(MASTER_IN)
idx = pd.DatetimeIndex(pd.to_datetime(m["datetime_utc"], utc=True))   # tz-aware UTC
pooled_valid = int(m["openaq_pm25"].notna().sum())

# --- single-sensor hourly series, floored to the hour, mean over any hour collisions ---
s = pd.read_parquet(SENSOR)
st = pd.to_datetime(s["datetime_utc"], utc=True, errors="coerce").dt.floor("h")
sv = pd.to_numeric(s["value"], errors="coerce")
sensor = pd.Series(sv.values, index=st).dropna()
sensor = sensor.groupby(level=0).mean()                    # 0 collisions expected
sensor.index = sensor.index.as_unit("us")                  # match master resolution

# --- align onto the master index; NaN where 24434 silent ---
aligned = sensor.reindex(idx)                              # tz-aware match
m["openaq_pm25"] = aligned.values                          # swap, in place, only this col

single_valid = int(m["openaq_pm25"].notna().sum())
ss = pd.Series(m["openaq_pm25"].values, index=idx)
v = ss.dropna()

# --- save new files; original untouched ---
m.to_parquet(OUT_PARQUET, index=False)
m.to_csv(OUT_CSV, index=False)

# --- report ---
print("pooled valid hours (old) :", pooled_valid)
print("single valid hours (new) :", single_valid, "(expect ~47656; sensor readings within master range)")
print("new last valid           :", v.index.max(), "(expect 2025-03-24 12:00)")
print("new mean                 :", round(float(v.mean()), 2))
print("new 2025 valid hours     :", int((v.index.year == 2025).sum()), "(expect ~1966, ending March)")
print("per-year valid (new):")
print(v.groupby(v.index.year).size().to_string())
print("columns unchanged        :", list(m.columns) == list(pd.read_parquet(MASTER_IN).columns))
print("wrote:", OUT_PARQUET, "and", OUT_CSV)
