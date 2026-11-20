"""R14 - small supporting numbers quoted in the revised text: yearly longest gap in valid days,
macro-F1 over the classes present, and correlation of day-t PM2.5 with ERA5 features."""
import numpy as np, pandas as pd
from _common import *
from sklearn.metrics import f1_score
from src.features.daily import EPA_LABELS, to_aqi

D = load(); feat = D["feat"]; o = {}
vd = feat.index[feat.valid]; gaps = pd.DataFrame({"start_last_valid": vd[:-1], "next_valid": vd[1:], "missing_days": (vd[1:] - vd[:-1]).days - 1})
g = gaps.loc[gaps.groupby(gaps.start_last_valid.dt.year).missing_days.idxmax()]
g.to_csv(OUT / "r14_longest_gap_by_year.csv", index=False)
o["longest_gap_by_year"] = [{"year": int(r.start_last_valid.year), "last_valid": str(r.start_last_valid.date()), "next_valid": str(r.next_valid.date()), "missing_days": int(r.missing_days)} for r in g.itertuples()]
m = D["X"].index.month; o["model_rows_by_month"] = pd.Series(m).value_counts().sort_index().to_dict()
P = pd.read_csv(OUT / "r03_holdout_predictions.csv", index_col=0, parse_dates=True)
t = to_aqi(pd.Series(P.y.values)).astype(str)
for k in ("xgboost", "persistence", "lasso"):
    c = to_aqi(pd.Series(P[k].values)).astype(str)
    o[f"macro_f1_6_labels_{k}"] = float(f1_score(t, c, labels=EPA_LABELS, average="macro", zero_division=0))
    o[f"macro_f1_5_present_{k}"] = float(f1_score(t, c, labels=EPA_LABELS[1:], average="macro", zero_division=0))
    o[f"accuracy_{k}"] = float((t == c).mean())
C = D["X_tr"].corr().abs().loc["pm25_t"]; o["abs_r_pm25_t_with_era5_gt_0.7"] = C[[c for c in C.index if c.startswith("era5_")]][lambda s: s > 0.7].round(3).to_dict()
dump("r14_extra_checks.json", o); print(json.dumps(o, indent=1))
