"""R12 - timing audit of the FIRMS features.
FIRMS daily counts are keyed by UTC acquisition date and broadcast to UTC hours (src/ingest/merge.py),
while daily features are aggregated over LOCAL (UTC+6) days with max() (src/features/daily.py).
A local day t therefore mixes UTC date t-1 (00:00-05:59 local) and UTC date t (06:00-23:59 local),
and UTC date t runs to 05:59 local on day t+1. This script documents that and re-fits without lag-0 FIRMS."""
import numpy as np, pandas as pd
from _common import *
from src.evaluate.metrics import reg_metrics

D = load(); df = D["df"]; t = pd.to_datetime(df["datetime_utc"], utc=True)
u = df.groupby(t.dt.tz_localize(None).dt.normalize())["firms_count"].max()          # value per UTC date
loc = D["feat"]["firms_count_lag0"].dropna()
idx = loc.index.intersection(u.index[1:])
prev = u.shift(1)
o = {"n_local_days_checked": len(idx),
     "lag0_equals_max_of_utc_t_minus_1_and_utc_t": int(np.isclose(loc[idx], np.maximum(u[idx], prev[idx])).sum()),
     "lag0_equals_utc_t_only": int(np.isclose(loc[idx], u[idx]).sum()),
     "lag0_equals_utc_t_minus_1_only": int(np.isclose(loc[idx], prev[idx]).sum()),
     "note": "UTC date t = 06:00 local on day t to 05:59 local on day t+1"}
f = D["features"]; y = D["y_te"].values; per = D["base"]["persistence"]
full = fit_xgb(D["X_tr"], D["y_tr"], D["X_te"])[1]
sets = {"drop_firms_lag0 (43 features)": [c for c in f if not c.endswith("_lag0")],
        "drop_all_firms (39 features)": [c for c in f if not c.startswith("firms_")],
        "drop_firms_and_era5 (14 features: PM history + calendar)": [c for c in f if not c.startswith(("firms_", "era5_", "wind_"))]}
o["refits"] = {"full_45": {**reg_metrics(y, full), "dm_p_vs_persistence": dm_test(y, full, per)["p_two_sided"]}}
for k, c in sets.items():
    p = fit_xgb(D["X_tr"][c], D["y_tr"], D["X_te"][c])[1]
    o["refits"][k] = {**reg_metrics(y, p), "dm_p_vs_persistence": dm_test(y, p, per)["p_two_sided"], "dm_p_vs_full": dm_test(y, p, full)["p_two_sided"]}
dump("r12_firms_timing.json", o); print(json.dumps(o, indent=1))
