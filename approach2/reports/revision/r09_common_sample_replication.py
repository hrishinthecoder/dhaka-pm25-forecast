"""R09 - Approach 1 (frozen LightGBM bundle) and Approach 2 (frozen XGBoost) scored on the
SAME holdout days, with Diebold-Mariano tests. Read-only on both frozen artifacts.
Run from approach2/. Approach 1 code is imported from ../approach1/src."""
import sys, joblib, numpy as np, pandas as pd, yaml
from _common import *
sys.path.insert(0, "../approach1/src")
from validation import split_final_holdout


def predict_bundle(bundle, X):
    """Same steps as approach1/src/models_tree.predict_bundle (median impute, anomaly flag, predict);
    re-stated here only to avoid importing optuna, which that module needs for tuning."""
    X = X[bundle["feature_cols"]]
    X_imp = X.fillna(bundle["medians"]).fillna(0.0)
    flag = (bundle["iso"].predict(X_imp) == -1).astype(int) if bundle["iso"] is not None else 0
    return bundle["model"].predict(X_imp.assign(anomaly_flag=flag))


cfg1 = yaml.safe_load(open("../approach1/config/config.yaml"))
tab = pd.read_parquet("../approach1/processed/supervised_daily.parquet")
dev, ho = split_final_holdout(tab, cfg1)
b = joblib.load("../approach1/artifacts/model_lightgbm.joblib")
p1 = pd.Series(predict_bundle(b, ho), index=ho.index); y1 = ho["y_next_day"]; per1 = ho["pm25_lag_1"]
o = {"approach1_own_holdout": {"n": len(ho), "first": ho.index.min().date(), "last": ho.index.max().date(), "n_dev": len(dev),
                               "lightgbm_rmse": rmse(y1, p1), "lightgbm_r2": r2(y1, p1), "persistence_rmse": rmse(y1, per1),
                               "dm_vs_persistence": dm_test(y1, p1, per1)}}
D = load(); P2 = pd.read_csv(OUT / "r03_holdout_predictions.csv", index_col=0, parse_dates=True)
common = ho.index.intersection(P2.index)
o["n_common_days"] = len(common); o["approach2_only_days"] = [str(d.date()) for d in P2.index.difference(ho.index)]
o["approach1_only_days"] = len(ho.index.difference(P2.index))
y = P2.loc[common, "y"].values; a1 = p1[common].values; a2 = P2.loc[common, "xgboost"].values; per = P2.loc[common, "persistence"].values
o["targets_identical_on_common_days"] = bool(np.allclose(y, y1[common].values, atol=1e-6)); o["max_abs_target_diff"] = float(np.max(np.abs(y - y1[common].values)))
o["common"] = {"persistence_rmse": rmse(y, per), "approach1_lightgbm_rmse": rmse(y, a1), "approach1_lightgbm_r2": r2(y, a1),
               "approach2_xgboost_rmse": rmse(y, a2), "approach2_xgboost_r2": r2(y, a2),
               "approach1_lightgbm_vs_persistence": dm_test(y, a1, per), "approach2_xgboost_vs_persistence": dm_test(y, a2, per),
               "approach1_lightgbm_vs_approach2_xgboost": dm_test(y, a1, a2), "pred_correlation": float(np.corrcoef(a1, a2)[0, 1]),
               "mean_abs_pred_difference": float(np.mean(np.abs(a1 - a2)))}
# Approach 1 days not in Approach 2 (pairs lacking a complete 7-day PM history)
extra = ho.index.difference(P2.index)
o["approach1_extra_days"] = {"n": len(extra), "lightgbm_rmse": rmse(y1[extra], p1[extra]), "persistence_rmse": rmse(y1[extra], per1[extra])}
# does Approach 1's daily series carry the 985 hours? compare its day-t means with Approach 2's on dev days
j = pd.concat([tab["pm25_lag_1"].rename("a1"), D["feat"]["pm25_mean"].rename("a2")], axis=1).dropna()
o["daily_mean_max_abs_diff_a1_vs_a2"] = float((j.a1 - j.a2).abs().max()); o["n_days_daily_mean_differs"] = int(((j.a1 - j.a2).abs() > 1e-6).sum())
o["approach1_features"] = {"n": len(b["feature_cols"]), "note": "plus anomaly_flag (IsolationForest, fit on dev rows) appended inside the bundle",
                           "holdout_rows_flagged_anomalous": int((b["iso"].predict(ho[b["feature_cols"]].fillna(b["medians"]).fillna(0.0)) == -1).sum()) if b["iso"] is not None else None}
pd.DataFrame({"y": y, "approach1_lightgbm": a1, "approach2_xgboost": a2, "persistence": per}, index=common).to_csv(OUT / "r09_common_sample_predictions.csv")
dump("r09_common_sample_replication.json", o); print(json.dumps(o, indent=1, default=str))
