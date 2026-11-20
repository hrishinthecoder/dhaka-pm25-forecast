"""R10 - the 45 model features with definition, source, timing and real-world latency.
Timing is read from src/features/daily.py; latency statements are from the providers' pages
(Copernicus C3S: ERA5 daily updates about 5 days behind real time; NASA FIRMS: near-real-time
detections within 3 hours of the satellite observation, standard product after 2-3 months)."""
import math, pandas as pd
from _common import *

f = json.load(open("models/final/feature_list.json"))
ERA5 = ("ERA5 reanalysis, hourly, grid point nearest 23.8103 N, 90.4125 E (Open-Meteo archive; boundary-layer height Jan-Jun 2024 from Copernicus CDS)",
        "about 5 days (ERA5T preliminary release); not available at the end of day t",
        "operational NWP analysis or short-range forecast for day t")
FIRMS = ("NASA FIRMS VIIRS S-NPP 375 m active fires (standard-processing archive), box of +/-100 km around the city centre",
         "standard product 2-3 months; near-real-time product within 3 hours of overpass",
         "FIRMS VIIRS near-real-time product (VNP14IMGTDL_NRT)")
PM = ("OpenAQ sensor 24434 (US Diplomatic Post, reference grade), hourly", "needs a live feed from the monitor; the public record ends 24 Mar 2025",
      "a live reference-grade PM2.5 monitor at the forecast location")
CAL = ("calendar (computed)", "none; known in advance", "not needed")
AGG = {"mean": "daily mean of hourly values", "min": "daily minimum of hourly values", "max": "daily maximum of hourly values", "sum": "daily sum of hourly values"}
rows = []
for c in f:
    g = group_of(c)
    if c == "pm25_t": d, t, s = "daily mean PM2.5 of day t (at least 18 hourly values)", "day t", PM
    elif c.startswith("pm25_lag"): k = c[-1]; d, t, s = f"daily mean PM2.5 of day t-{k} (any day with at least one hourly value)", f"day t-{k}", PM
    elif c.startswith("pm25_roll"):
        w = c[-1]; st = "mean" if "mean" in c else "standard deviation"
        d, t, s = f"{st} of daily mean PM2.5 over days t-{int(w)-1} to t (all {w} days required)", f"days t-{int(w)-1}..t", PM
    elif c.startswith("wind_"): d, t, s = f"daily mean of the {'eastward (u)' if c=='wind_u_mean' else 'northward (v)'} 10 m wind component, from hourly speed and direction", "day t", ERA5
    elif c.startswith("era5_"):
        var, a = c[5:].rsplit("_", 1); d, t, s = f"{var.replace('_', ' ')}: {AGG[a]}", "day t", ERA5
    elif c.startswith("firms_"):
        k = int(c[-1]); what = "number of fire detections" if "count" in c else "summed fire radiative power (MW)"
        day = lambda n: "t" if n == 0 else f"t-{n}"
        d = f"{what}, larger of UTC dates {day(k+1)} and {day(k)} (see timing)"
        t = (f"UTC dates {day(k+1)} and {day(k)}; UTC date {day(k)} ends 05:59 local on day {day(k-1)}" if k > 0
             else "UTC dates t-1 and t; UTC date t ends 05:59 local on day t+1 (about 6 h after the end of local day t)")
        s = FIRMS
    else:
        d = {"day_of_week": "day of week of day t (0 = Monday)", "month": "calendar month of day t", "is_weekend": "weekend flag for day t, from the upstream calendar column",
             "is_holiday": "Bangladesh public holiday on day t", "is_ramadan": "day t falls in Ramadan (fixed table of approximate dates, 2016-2026)"}[c]
        t, s = "day t (local)", CAL
    rows.append({"no": len(rows) + 1, "feature": c, "group": g, "definition": d, "refers_to": t, "source": s[0], "latency_in_practice": s[1], "near_real_time_substitute": s[2]})
T = pd.DataFrame(rows); T.to_csv(OUT / "r10_feature_table.csv", index=False)
lat, lon, r = 23.8103, 90.4125, 100.0
dlat = r / 111.0; dlon = r / (111.0 * math.cos(math.radians(lat)))
df = pd.read_parquet(CFG["data"]["input_path"], columns=["datetime_local", "is_weekend", "day_of_week"])
wk = df.groupby("day_of_week").is_weekend.max().to_dict()
dump("r10_feature_table_notes.json", {"firms_box_west_south_east_north": [round(lon - dlon, 4), round(lat - dlat, 4), round(lon + dlon, 4), round(lat + dlat, 4)],
     "firms_box_size_km": [200.0, 200.0], "group_counts": T.group.value_counts().to_dict(), "is_weekend_by_day_of_week": wk})
print(T.group.value_counts().to_dict(), wk, round(lon - dlon, 4), round(lat - dlat, 4), round(lon + dlon, 4), round(lat + dlat, 4)); print(T[["feature","refers_to"]].iloc[[0,4,5,8,9,32,39,41]].to_string())
