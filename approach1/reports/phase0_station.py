"""
phase0_station.py — SINGLE-STATION VERIFICATION (the critical Phase 0 gate).

WHAT THIS DOES
    The merged master dropped all OpenAQ identifiers, so we cannot read single-station
    status off `openaq_pm25` directly. Instead we reverse-engineer how that column was
    built: we rebuild an hourly series from EACH raw pm25 sensor file, then test which
    construction reproduces the master's `openaq_pm25`:
        (A) a single dominant sensor (24434, US-embassy-area, by far the most data), or
        (B) a multi-sensor average / coalesce across different sites.

WHY IT MATTERS
    The whole study is framed as TEMPORAL forecasting at ONE point. If `openaq_pm25`
    is actually several geographically-spread sensors averaged together, that framing
    breaks and the build must STOP. (B) = stop-and-report; (A) = single-station holds.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
MASTER = BASE / "processed" / "dhaka_aq_master.parquet"
OPENAQ_DIR = BASE / "raw" / "openaq"
TOL = 0.05  # µg/m³ tolerance for "equal" (floats from averaging round-trips)

# --- master ground-truth series, hourly, keyed by UTC hour --------------------
m = pd.read_parquet(MASTER, columns=["datetime_utc", "openaq_pm25"])
m["hour"] = pd.to_datetime(m["datetime_utc"], utc=True).dt.floor("h")
m = m.dropna(subset=["openaq_pm25"]).set_index("hour")["openaq_pm25"]
m = m[~m.index.duplicated(keep="first")]
print("master non-null openaq_pm25 hours:", len(m))

# --- rebuild an hourly series for every pm25 sensor ---------------------------
files = sorted(OPENAQ_DIR.glob("sensor_*_pm25.parquet"))
sensor_hourly = {}   # sensor_id -> Series(value indexed by UTC hour)
meta = {}            # sensor_id -> (location_id, location_name, distance proxy)
for f in files:
    s = pd.read_parquet(f, columns=["datetime_utc", "value", "location_id",
                                     "location_name", "sensor_id"])
    sid = int(s["sensor_id"].iloc[0])
    hr = pd.to_datetime(s["datetime_utc"], utc=True).dt.floor("h")
    ser = pd.Series(s["value"].values, index=hr)
    ser = ser.groupby(level=0).mean()          # collapse any sub-hourly to hourly mean
    sensor_hourly[sid] = ser
    meta[sid] = (int(s["location_id"].iloc[0]), str(s["location_name"].iloc[0]))

# wide frame: one column per sensor, aligned on UTC hour
wide = pd.DataFrame(sensor_hourly)
print("union of sensor hours:", len(wide), " n sensors:", wide.shape[1])

# --- HYPOTHESIS A: dominant sensor 24434 alone --------------------------------
dom = 24434
a = wide[dom].reindex(m.index)
match_a = (np.abs(a - m) <= TOL)
print("\n[A] dominant sensor 24434 (loc %s '%s')" % meta[dom])
print("    coverage of master hours by 24434:", int(a.notna().sum()), "/", len(m))
print("    exact matches (|diff|<=%.2f):" % TOL, int(match_a.sum()),
      "= %.2f%% of master" % (100*match_a.sum()/len(m)))

# --- HYPOTHESIS B: row-wise mean across ALL sensors present that hour ---------
b = wide.mean(axis=1, skipna=True).reindex(m.index)
match_b = (np.abs(b - m) <= TOL)
print("\n[B] multi-sensor row-mean across all", wide.shape[1], "sensors")
print("    exact matches (|diff|<=%.2f):" % TOL, int(match_b.sum()),
      "= %.2f%% of master" % (100*match_b.sum()/len(m)))

# --- which single sensor matches each master hour? (priority/coalesce test) ---
# For every master hour, count sensors whose hourly value equals the master value.
aligned = wide.reindex(m.index)
eq = aligned.sub(m, axis=0).abs() <= TOL
n_match_per_hour = eq.sum(axis=1)
print("\n[per-hour] master hours matched by exactly one sensor:",
      int((n_match_per_hour == 1).sum()))
print("           matched by >=1 sensor:", int((n_match_per_hour >= 1).sum()),
      "= %.2f%% of master" % (100*(n_match_per_hour >= 1).sum()/len(m)))
print("           matched by NO sensor:", int((n_match_per_hour == 0).sum()))
print("           matched by >=2 sensors (ambiguous):",
      int((n_match_per_hour >= 2).sum()))

# Which sensor is the matching one, hour by hour -> distribution
which = eq.idxmax(axis=1).where(n_match_per_hour >= 1)
print("\n[source sensor distribution] for matched master hours (top 10):")
dist = which.value_counts().head(10)
for sid, n in dist.items():
    print(f"    sensor {int(sid):>10}  loc {meta[int(sid)][0]:>8}  n={n:>6}  '{meta[int(sid)][1][:40]}'")

# How many DISTINCT locations contribute matched hours?
matched_locs = set(meta[int(sid)][0] for sid in which.dropna().unique())
print("\ndistinct location_ids contributing to openaq_pm25:", len(matched_locs))
print("location_ids:", sorted(matched_locs))
