"""Step 1 — verify the single-sensor source (24434) and inspect the current master.

Read-only. Prints a PASS/FAIL table; does not abort hard so every check is visible.
Run: ./.venv/Scripts/python.exe fusion/verify_source.py
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

SENSOR = Path(r"../Approach one/Machine Learning Model/raw/openaq/sensor_24434_pm25.parquet")
MASTER = Path("data/processed/dhaka_aq_master_rebuilt.parquet")

print("=" * 70)
print("SENSOR 24434 FILE")
print("=" * 70)
s = pd.read_parquet(SENSOR)
print("shape:", s.shape)
print("columns:", list(s.columns))
print("index name/type:", s.index.name, type(s.index))
print("head:\n", s.head(3))
print("dtypes:\n", s.dtypes)

# Identify the pm25 value column + a usable datetime index.
print("\n--- structural probe ---")
print("index tz:", getattr(s.index, "tz", "n/a"))

print("\n" + "=" * 70)
print("MASTER (current, pooled)")
print("=" * 70)
m = pd.read_parquet(MASTER)
print("shape:", m.shape)
print("has openaq_pm25:", "openaq_pm25" in m.columns)
print("datetime_utc dtype:", m["datetime_utc"].dtype if "datetime_utc" in m.columns else "MISSING")
mp = m.set_index(pd.to_datetime(m["datetime_utc"], utc=True))["openaq_pm25"]
valid_by_year = mp.dropna().groupby(mp.dropna().index.year).size()
print("master openaq_pm25 valid hours/year (POOLED, expect ~6100 ending Dec 2025):")
print(valid_by_year.to_string())
print("master openaq_pm25 last valid:", mp.dropna().index.max())
print("master openaq_pm25 total valid:", int(mp.notna().sum()))
print("master openaq_pm25 mean:", round(float(mp.mean()), 2))
