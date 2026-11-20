"""R00 - reproduce the frozen v2.0 headline numbers (read-only on frozen artifacts).

Run from approach2/:  python reports/revision/r00_reproduce_frozen.py
Writes reports/revision/r00_reproduce_frozen.json only.
"""
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, yaml, xgboost
sys.path.insert(0, ".")
from src.features.daily import build_daily, model_matrix, split_masks, holdout_start
from src.models.train import baselines, fit_scaler, apply_scaler
from src.evaluate.metrics import reg_metrics, bootstrap_ci, diebold_mariano, paired_bootstrap_diff
from xgboost import XGBRegressor

cfg = yaml.safe_load(Path("config/model.yaml").read_text())
df = pd.read_parquet(cfg["data"]["input_path"])
feat, features = build_daily(df, cfg)
X, y = model_matrix(feat, features)
tr, va, te = split_masks(X.index, cfg)
X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
frozen_feats = json.load(open("models/final/feature_list.json"))
out = {"xgboost_version": xgboost.__version__,
       "n_pairs": int(len(X)), "n_train": int(tr.sum()), "n_holdout": int(te.sum()),
       "n_features": len(features), "features_match_frozen": features == frozen_feats,
       "train_first": str(X_tr.index.min().date()), "train_last": str(X_tr.index.max().date()),
       "holdout_first": str(X_te.index.min().date()), "holdout_last": str(X_te.index.max().date())}

paired = feat[feat["pair_valid"]].dropna(subset=features + ["y_next"])
hs = holdout_start(X.index, cfg)
base = baselines(paired[paired.index < hs], paired[paired.index >= hs])

# (a) frozen artifact
scaler = joblib.load("models/final/scaler.joblib")
frozen = joblib.load("models/final/xgboost_final.joblib")
p_frozen = frozen.predict(scaler.transform(X_te.values))
# (b) refit with frozen hyperparameters under the installed xgboost
hp = json.load(open("models/final/hyperparameters.json"))
sc2 = fit_scaler(X_tr)
refit = XGBRegressor(**hp["tuned"], **hp["fixed"]).fit(apply_scaler(sc2, X_tr), y_tr.values)
p_refit = refit.predict(apply_scaler(sc2, X_te))
out["scaler_refit_matches_frozen"] = bool(np.allclose(sc2.mean_, scaler.mean_) and np.allclose(sc2.scale_, scaler.scale_))
out["max_abs_pred_diff_frozen_vs_refit"] = float(np.max(np.abs(p_frozen - p_refit)))

for name, p in {"xgboost_frozen": p_frozen, "xgboost_refit": p_refit,
                "persistence": base["persistence"], "climatology": base["climatology"]}.items():
    out[name] = {**reg_metrics(y_te.values, p), **bootstrap_ci(y_te.values, p, 1000, 42)}
for name, p in {"xgboost_frozen": p_frozen, "xgboost_refit": p_refit}.items():
    out[name + "_vs_persistence"] = {**paired_bootstrap_diff(y_te.values, p, base["persistence"], 1000, 42),
                                     **diebold_mariano(y_te.values, p, base["persistence"])}
Path("reports/revision").mkdir(parents=True, exist_ok=True)
json.dump(out, open("reports/revision/r00_reproduce_frozen.json", "w"), indent=2)
print(json.dumps(out, indent=2))
