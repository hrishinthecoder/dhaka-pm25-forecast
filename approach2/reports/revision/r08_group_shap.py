"""R08 - group-level SHAP for the frozen primary XGBoost model.
Exact TreeSHAP from xgboost (pred_contribs=True); no extra dependency.
Shares are of summed mean |SHAP|; reported on holdout rows and on training rows."""
import joblib, numpy as np, pandas as pd, xgboost as xgb
from _common import *

D = load(); f = D["features"]
model = joblib.load("models/final/xgboost_final.joblib"); scaler = joblib.load("models/final/scaler.joblib")
o = {}; tabs = {}
for name, X in (("holdout", D["X_te"]), ("train", D["X_tr"])):
    c = model.get_booster().predict(xgb.DMatrix(scaler.transform(X.values)), pred_contribs=True)
    assert np.allclose(c.sum(1), model.predict(scaler.transform(X.values)), atol=1e-2)
    s = pd.Series(np.abs(c[:, :-1]).mean(0), index=f)
    g = s.groupby([group_of(k) for k in f]).sum()
    o[name] = {"n": len(X), "group_sum_mean_abs_shap": g.round(3).to_dict(), "group_share_pct": (100 * g / g.sum()).round(1).to_dict(),
               "top10": s.sort_values(ascending=False).head(10).round(3).to_dict()}
    # signed group contribution per day -> share of variance of the prediction explained by each group's summed SHAP
    gs = pd.DataFrame(c[:, :-1], columns=f).T.groupby([group_of(k) for k in f]).sum().T
    o[name]["mean_abs_of_group_summed_shap"] = gs.abs().mean().round(3).to_dict()
    tabs[name] = s
pd.DataFrame(tabs).assign(group=[group_of(k) for k in f]).to_csv(OUT / "r08_shap_per_feature.csv")
old = pd.read_csv("reports/shap_importance.csv", index_col=0); o["frozen_shap_importance_csv_top5"] = old.iloc[:5, 0].round(3).to_dict()
dump("r08_group_shap.json", o); print(json.dumps(o, indent=1))
