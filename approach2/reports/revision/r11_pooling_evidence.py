"""R11 - station-composition evidence for the pooled (v1.0) target, from files in this repository:
approach1/processed/dhaka_aq_master.parquet (pooled hourly openaq_pm25) versus
approach2/data/processed/dhaka_aq_master_singlestation.parquet (sensor 24434 only).
No v1.0 skill numbers are computed. Per-sensor counts (number of sensors per hour, the
Jahangirnagar overlap, distances) need the raw per-sensor OpenAQ files, which are not in the repository."""
import numpy as np, pandas as pd, yaml
from _common import *

a = pd.read_parquet("../approach1/processed/dhaka_aq_master.parquet", columns=["datetime_utc", "openaq_pm25"])
b = pd.read_parquet(CFG["data"]["input_path"], columns=["datetime_utc", "openaq_pm25"])
a = a.set_index(pd.to_datetime(a.datetime_utc, utc=True)).openaq_pm25.dropna(); b = b.set_index(pd.to_datetime(b.datetime_utc, utc=True)).openaq_pm25.dropna()
j = pd.concat([a.rename("pooled"), b.rename("s24434")], axis=1).dropna(subset=["pooled"])
last = b.index.max()
o = {"pooled_hours": int(len(j)), "pooled_first": j.index.min(), "pooled_last": j.index.max(), "s24434_hours": int(len(b)), "s24434_first": b.index.min(), "s24434_last": last,
     "pooled_hours_with_24434_reporting": int(j.s24434.notna().sum()),
     "pooled_equals_24434": int(np.isclose(j.pooled, j.s24434).sum()),
     "pooled_differs_from_24434_while_24434_reports": int((j.s24434.notna() & ~np.isclose(j.pooled, j.s24434)).sum()),
     "pooled_hours_without_24434": int(j.s24434.isna().sum()),
     "pooled_hours_before_24434_first": int((j.index < b.index.min()).sum()),
     "pooled_hours_after_24434_last": int((j.index > last).sum()),
     "pooled_hours_without_24434_inside_its_span": int((j.s24434.isna() & (j.index >= b.index.min()) & (j.index <= last)).sum())}
by = j.groupby(j.index.year).agg(pooled_hours=("pooled", "size"), with_24434=("s24434", lambda s: s.notna().sum()))
by["pct_pooled_hours_including_24434"] = (100 * by.with_24434 / by.pooled_hours).round(1)
by["pct_pooled_hours_equal_to_24434"] = (100 * j.assign(eq=np.isclose(j.pooled, j.s24434)).groupby(j.index.year).eq.mean()).round(1)
by.to_csv(OUT / "r11_pooled_share_by_year.csv")
d = j.loc[j.s24434.notna() & ~np.isclose(j.pooled, j.s24434)]
o["where_pooled_differs"] = {"n": len(d), "first": d.index.min(), "last": d.index.max(), "mean_pooled": float(d.pooled.mean()), "mean_24434": float(d.s24434.mean()),
                             "ratio_of_means": float(d.pooled.mean() / d.s24434.mean()), "pearson_r": float(d.pooled.corr(d.s24434))}
loc = j.pooled.copy(); loc.index = loc.index.tz_convert("Asia/Dhaka").tz_localize(None)
dd = loc.groupby(loc.index.normalize()).count(); after = dd[dd.index > last.tz_convert("Asia/Dhaka").tz_localize(None).normalize()]
o["pooled_valid_days_ge18h_after_24434_last_day"] = int((after >= 18).sum())
st = yaml.safe_load(open("config/stations_resolved.yaml"))["stations"]
def km(la, lo, la0=23.8103, lo0=90.4125):
    p = np.pi / 180; x = (lo - lo0) * p * np.cos((la + la0) / 2 * p); yy = (la - la0) * p
    return 6371 * np.sqrt(x * x + yy * yy)
S = pd.DataFrame([{"location_id": s["id"], "name": s["name"], "parameters": ",".join(s["parameters"]), "has_pm25": "pm25" in s["parameters"],
                   "km_from_centre": round(km(s["lat"], s["lon"]), 1)} for s in st])
S.to_csv(OUT / "r11_stations_resolved.csv", index=False)
o["stations_resolved_yaml"] = {"n_locations": len(S), "n_locations_with_pm25": int(S.has_pm25.sum()), "km_min": float(S.km_from_centre.min()), "km_max": float(S.km_from_centre.max())}
dump("r11_pooling_evidence.json", o); print(json.dumps(o, indent=1, default=str)); print(by); print(S.to_string(index=False))
