"""R16 - station composition of the pooled (v1.0) PM2.5 target, from the raw per-sensor OpenAQ files.

Needs the raw download (per-sensor parquet files, _sensors.csv, _locations.json), which is not
redistributed in this repository:
    python reports/revision/r16_pooling_from_raw.py [raw_dir]        (default ../approach1/raw)
Pooling rule reproduced here: hourly mean across all PM2.5 sensors reporting in that hour.
No forecast-skill numbers for the pooled design are computed."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from _common import OUT, dump

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "../approach1/raw")
sens = pd.read_csv(RAW / "_sensors.csv"); locs = json.load(open(RAW / "_locations.json"))
files = sorted(glob.glob(str(RAW / "sensor_*.parquet")))
o = {"locations_listed": len(locs), "locations_in_sensor_table": int(sens.location_id.nunique()),
     "sensors_listed": int(len(sens)), "sensors_listed_by_parameter": sens.parameter.value_counts().to_dict(),
     "data_files": len(files), "data_files_by_parameter": pd.Series([Path(f).stem.split("_")[-1] for f in files]).value_counts().to_dict()}
# hourly series per PM2.5 sensor
S = {}
for f in files:
    sid, par = Path(f).stem.split("_")[1:3]
    if par != "pm25": continue
    d = pd.read_parquet(f)
    t = pd.to_datetime(d["datetime_utc"], utc=True, errors="coerce").dt.floor("h")
    s = pd.Series(pd.to_numeric(d["value"], errors="coerce").values, index=t).dropna()
    S[int(sid)] = s.groupby(level=0).mean()
W = pd.DataFrame(S).sort_index()
listed_pm = set(sens[sens.parameter == "pm25"].sensor_id)
o["pm25_sensors_listed"] = len(listed_pm); o["pm25_sensors_with_data_file"] = len(S)
o["pm25_sensors_with_at_least_one_value"] = int((W.notna().sum() > 0).sum())
o["pm25_sensors_listed_without_data"] = sorted(int(x) for x in listed_pm - {k for k in S if W[k].notna().any()})
o["locations_with_pm25_data"] = int(sens[sens.sensor_id.isin([k for k in S if W[k].notna().any()])].location_id.nunique())
# per-sensor table with OpenAQ metadata
meta = {l["id"]: l for l in locs}
rows = []
for _, r in sens.iterrows():
    l = meta.get(r.location_id, {}); s = W[r.sensor_id].dropna() if r.sensor_id in W else pd.Series(dtype=float)
    rows.append({"location_id": r.location_id, "location_name": r.location_name, "km_from_centre": round(r.distance_km, 1), "sensor_id": r.sensor_id, "parameter": r.parameter,
                 "isMonitor": l.get("isMonitor"), "isMobile": l.get("isMobile"), "provider": (l.get("provider") or {}).get("name"), "owner": (l.get("owner") or {}).get("name"),
                 "instrument": "; ".join(i.get("name", "") for i in (l.get("instruments") or [])),
                 "pm25_hours": int(len(s)), "first_hour_utc": s.index.min() if len(s) else None, "last_hour_utc": s.index.max() if len(s) else None,
                 "mean_pm25": round(float(s.mean()), 1) if len(s) else None})
T = pd.DataFrame(rows); T.to_csv(OUT / "r16_sensor_inventory.csv", index=False)
pm = T[(T.parameter == "pm25") & (T.pm25_hours > 0)]
o["pm25_data_sensors_isMonitor_true"] = int((pm.isMonitor == True).sum()); o["pm25_data_sensors_isMonitor_false"] = int((pm.isMonitor == False).sum())
o["pm25_data_sensors_by_provider"] = pm.provider.value_counts().to_dict(); o["pm25_data_sensors_by_instrument"] = pm.instrument.value_counts().to_dict()
o["km_range_pm25_data_sensors"] = [float(pm.km_from_centre.min()), float(pm.km_from_centre.max())]; o["km_range_all_locations"] = [float(T.km_from_centre.min()), float(T.km_from_centre.max())]
# pooled series
n = W.notna().sum(axis=1); pooled = W.mean(axis=1)[n > 0]; n = n[n > 0]; ref = W[24434]
last = ref.dropna().index.max(); first = ref.dropna().index.min()
o.update({"pooled_hours": int(len(pooled)), "pooled_first": pooled.index.min(), "pooled_last": pooled.index.max(),
          "s24434_hours": int(ref.notna().sum()), "s24434_first": first, "s24434_last": last,
          "pooled_hours_from_24434_alone": int(((n == 1) & ref.reindex(n.index).notna()).sum()),
          "pooled_hours_with_two_or_more_sensors": int((n >= 2).sum()), "max_sensors_in_one_hour": int(n.max()),
          "hour_of_max_sensors": n.idxmax(), "sensors_per_hour_distribution": n.value_counts().sort_index().to_dict(),
          "pooled_hours_including_24434": int(ref.reindex(n.index).notna().sum()),
          "pooled_hours_before_24434_first": int((n.index < first).sum()), "pooled_hours_after_24434_last": int((n.index > last).sum())})
has = ref.reindex(n.index).notna()
by = pd.DataFrame({"pooled_hours": n.groupby(n.index.year).size(), "hours_including_24434": has.groupby(has.index.year).sum(),
                   "mean_sensors_per_hour": n.groupby(n.index.year).mean().round(2), "max_sensors_per_hour": n.groupby(n.index.year).max(),
                   "distinct_sensors_reporting": W.notna().groupby(W.index.year).any().sum(axis=1)})
by["pct_hours_including_24434"] = (100 * by.hours_including_24434 / by.pooled_hours).round(1); by.to_csv(OUT / "r16_pooled_share_by_year.csv")
loc = pooled.copy(); loc.index = loc.index.tz_convert("Asia/Dhaka").tz_localize(None); cnt = loc.groupby(loc.index.normalize()).count()
lastday = last.tz_convert("Asia/Dhaka").tz_localize(None).normalize()
o["pooled_valid_days_ge18h_total"] = int((cnt >= 18).sum()); o["pooled_valid_days_ge18h_after_24434_last_day"] = int((cnt[cnt.index > lastday] >= 18).sum())
# overlap of every other sensor with 24434
ov = []
for k in W.columns:
    if k == 24434: continue
    j = pd.concat([ref.rename("ref"), W[k].rename("x")], axis=1).dropna()
    if len(j) == 0:
        ov.append({"sensor_id": k, "overlap_hours": 0}); continue
    ov.append({"sensor_id": k, "location_name": T.set_index("sensor_id").location_name[k], "km_from_centre": T.set_index("sensor_id").km_from_centre[k],
               "overlap_hours": len(j), "first": j.index.min(), "last": j.index.max(), "mean_24434": round(j.ref.mean(), 1), "mean_sensor": round(j.x.mean(), 1),
               "ratio_of_means": round(j.x.mean() / j.ref.mean(), 3), "pearson_r": round(j.ref.corr(j.x), 3) if len(j) > 2 else None,
               "mean_abs_diff": round((j.x - j.ref).abs().mean(), 1)})
O = pd.DataFrame(ov).sort_values("overlap_hours", ascending=False); O.to_csv(OUT / "r16_overlap_with_24434.csv", index=False)
o["sensors_overlapping_24434"] = int((O.overlap_hours > 0).sum())
# check against the pooled master shipped in the repository
a = pd.read_parquet("../approach1/processed/dhaka_aq_master.parquet", columns=["datetime_utc", "openaq_pm25"])
a = a.set_index(pd.to_datetime(a.datetime_utc, utc=True)).openaq_pm25.dropna()
j = pd.concat([a.rename("master"), pooled.rename("rebuilt")], axis=1)
o["repo_master_hours"] = int(len(a)); o["hours_in_both"] = int(j.dropna().shape[0]); o["hours_only_in_rebuilt"] = int((j.master.isna() & j.rebuilt.notna()).sum())
o["hours_only_in_master"] = int((j.master.notna() & j.rebuilt.isna()).sum()); o["max_abs_diff_where_both"] = float((j.master - j.rebuilt).abs().max())
dump("r16_pooling_from_raw.json", o); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(json.dumps(o, indent=1, default=str)); print(by); print(T.drop(columns=["location_id", "owner", "isMobile"]).to_string(index=False)); print(O.to_string(index=False))
