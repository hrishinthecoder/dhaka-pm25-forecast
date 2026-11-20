"""R02 - value-QC sensitivity: mask hourly PM2.5 >= 900 (985 fault code), rebuild, retrain.

Frozen hyperparameters, same 18-hour rule, same anchored 12-month holdout rule.
Nothing frozen is modified; the masked data exist only in memory.
"""
import numpy as np, pandas as pd
from _common import *
from src.evaluate.metrics import reg_metrics

base = load()
_, p0, _ = fit_xgb(base["X_tr"], base["y_tr"], base["X_te"])
df = base["df"].copy()
hi = df["openaq_pm25"] >= 900
vals = df.loc[hi, ["datetime_local", "openaq_pm25"]]
vals.to_csv(OUT / "r02_hours_ge900.csv", index=False)
df.loc[hi, "openaq_pm25"] = np.nan
q = load(df)
_, p1, _ = fit_xgb(q["X_tr"], q["y_tr"], q["X_te"])
o = {"hours_masked": int(hi.sum()),
     "original": {"n_pairs": len(base["X"]), "n_train": int(base["tr"].sum()), "n_holdout": int(base["te"].sum()),
                  "xgboost": reg_metrics(base["y_te"], p0), "persistence": reg_metrics(base["y_te"], base["base"]["persistence"]),
                  "dm_vs_persistence": dm_test(base["y_te"], p0, base["base"]["persistence"])},
     "masked": {"n_pairs": len(q["X"]), "n_train": int(q["tr"].sum()), "n_holdout": int(q["te"].sum()),
                "holdout_first": q["X_te"].index.min().date(), "holdout_last": q["X_te"].index.max().date(),
                "xgboost": reg_metrics(q["y_te"], p1), "persistence": reg_metrics(q["y_te"], q["base"]["persistence"]),
                "dm_vs_persistence": dm_test(q["y_te"], p1, q["base"]["persistence"])}}
o["holdout_days_identical"] = bool(base["X_te"].index.equals(q["X_te"].index))
o["holdout_targets_identical"] = bool(np.allclose(base["y_te"].values, q["y_te"].values)) if o["holdout_days_identical"] else False
o["holdout_features_identical"] = bool(np.allclose(base["X_te"].values, q["X_te"].values)) if o["holdout_days_identical"] else False
o["max_abs_holdout_pred_change"] = float(np.max(np.abs(p0 - p1))) if o["holdout_days_identical"] else None
o["mean_abs_holdout_pred_change"] = float(np.mean(np.abs(p0 - p1))) if o["holdout_days_identical"] else None
# effect on the 20 affected daily means
d0, d1 = base["feat"]["pm25_mean"], q["feat"]["pm25_mean"]
ch = (d0 - d1).dropna(); ch = ch[ch.abs() > 1e-9]
o["daily_means_changed"] = int(len(ch)); o["daily_mean_change_min_max"] = [float(ch.min()), float(ch.max())]
o["days_newly_invalid"] = int((base["feat"]["valid"] & ~q["feat"]["valid"]).sum())
dump("r02_qc985_sensitivity.json", o); print(json.dumps(o, indent=1, default=str)); print(vals.to_string())
