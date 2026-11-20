"""R07 - error by concentration range and season; AQI per-class recall and confusion matrix.
XGBoost = frozen model predictions; persistence on the same 219 holdout days."""
import numpy as np, pandas as pd
from _common import *
from src.features.daily import EPA_LABELS, to_aqi
from src.evaluate.metrics import season_of
from sklearn.metrics import confusion_matrix

P = pd.read_csv(OUT / "r03_holdout_predictions.csv", index_col=0, parse_dates=True)
y, x, p = P.y.values, P.xgboost.values, P.persistence.values
def row(lab, m):
    n = int(m.sum())
    if n == 0: return {"stratum": lab, "n": 0}
    ex, ep = x[m] - y[m], p[m] - y[m]
    r = {"stratum": lab, "n": n, "obs_mean": y[m].mean(), "xgb_bias": ex.mean(), "xgb_mae": np.abs(ex).mean(), "xgb_rmse": np.sqrt((ex ** 2).mean()),
         "per_bias": ep.mean(), "per_mae": np.abs(ep).mean(), "per_rmse": np.sqrt((ep ** 2).mean())}
    r["d_rmse"] = r["xgb_rmse"] - r["per_rmse"]; r["d_mae"] = r["xgb_mae"] - r["per_mae"]
    r["xgb_underpredicted_n"] = int((ex < 0).sum())
    r["dm_p"] = dm_test(y[m], x[m], p[m])["p_two_sided"] if n >= 20 else np.nan
    return r
bins = [(0, 35.4), (35.4, 55.4), (55.4, 125.4), (125.4, 225.4), (225.4, 1e9)]
rows = [row(f"observed {a:g}-{b:g}" if b < 1e8 else f"observed >{a:g}", (y > a) & (y <= b)) for a, b in bins]
rows += [row("observed >150", y > 150), row("observed >200", y > 200), row("observed >240", y > 240), row("observed <=150", y <= 150)]
q = np.quantile(y, [0, .25, .5, .75, 1]); rows += [row(f"observed quartile {i+1} ({q[i]:.0f}-{q[i+1]:.0f})", (y >= q[i]) & ((y < q[i+1]) | (i == 3))) for i in range(4)]
seas_t1 = np.array([season_of(m) for m in (P.index + pd.Timedelta(days=1)).month])
rows += [row(f"season (day t+1) {s}", seas_t1 == s) for s in ("winter_dry", "transition", "monsoon")]
# stratify by day-t level too (known at issue time, so not conditioned on the outcome)
pt = p; rows += [row("day-t PM2.5 <=55.4", pt <= 55.4), row("day-t PM2.5 55.4-125.4", (pt > 55.4) & (pt <= 125.4)), row("day-t PM2.5 >125.4", pt > 125.4)]
T = pd.DataFrame(rows); T.to_csv(OUT / "r07_error_by_stratum.csv", index=False)
o = {"max_observed": float(y.max()), "max_xgb_prediction": float(x.max()), "n_obs_gt_240": int((y > 240).sum()),
     "n_obs_gt_240_underpredicted": int(((y > 240) & (x < y)).sum()),
     "largest_miss": {"date_t": str(P.index[np.argmin(x - y)].date()), "observed": float(y[np.argmin(x - y)]), "xgb": float(x[np.argmin(x - y)]), "persistence": float(p[np.argmin(x - y)])},
     "share_of_sq_error_reduction_from_obs_gt_150": float((((p - y) ** 2 - (x - y) ** 2)[y > 150]).sum() / ((p - y) ** 2 - (x - y) ** 2).sum())}
# AQI classes
t = to_aqi(pd.Series(y)).astype(str); cx = to_aqi(pd.Series(x)).astype(str); cp = to_aqi(pd.Series(p)).astype(str)
cm = confusion_matrix(t, cx, labels=EPA_LABELS); cmp_ = confusion_matrix(t, cp, labels=EPA_LABELS)
pd.DataFrame(cm, index=[f"obs_{l}" for l in EPA_LABELS], columns=[f"pred_{l}" for l in EPA_LABELS]).to_csv(OUT / "r07_confusion_xgboost.csv")
pd.DataFrame(cmp_, index=[f"obs_{l}" for l in EPA_LABELS], columns=[f"pred_{l}" for l in EPA_LABELS]).to_csv(OUT / "r07_confusion_persistence.csv")
C = pd.DataFrame({"class": EPA_LABELS, "n_obs": cm.sum(1), "xgb_recall": np.diag(cm) / np.maximum(cm.sum(1), 1), "xgb_precision": np.diag(cm) / np.maximum(cm.sum(0), 1),
                  "n_pred_xgb": cm.sum(0), "per_recall": np.diag(cmp_) / np.maximum(cmp_.sum(1), 1)})
C.loc[C.n_obs == 0, ["xgb_recall", "per_recall"]] = np.nan
C.to_csv(OUT / "r07_aqi_per_class.csv", index=False)
o["aqi_accuracy_xgb"] = float(np.trace(cm) / cm.sum()); o["aqi_accuracy_persistence"] = float(np.trace(cmp_) / cmp_.sum())
dump("r07_extremes.json", o); pd.set_option("display.width", 250)
print(T.round(2).to_string(index=False)); print(o); print(C.round(3).to_string(index=False)); print(cm)
