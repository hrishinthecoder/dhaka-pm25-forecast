"""Shared loaders for the revision analyses (read-only on frozen artifacts).

All scripts are run from approach2/:  python reports/revision/<script>.py
Outputs go to reports/revision/ only. Requires xgboost==3.2.0 for exact refits.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, ".")
from src.features.daily import build_daily, model_matrix, split_masks, holdout_start
from src.models.train import baselines, fit_scaler, apply_scaler
from xgboost import XGBRegressor

OUT = Path("reports/revision"); OUT.mkdir(parents=True, exist_ok=True)
CFG = yaml.safe_load(Path("config/model.yaml").read_text())
HP = json.load(open("models/final/hyperparameters.json"))
SEED = 42

GROUPS = {
    "pm_history": lambda c: c.startswith("pm25_"),
    "era5": lambda c: c.startswith("era5_") or c.startswith("wind_"),
    "firms": lambda c: c.startswith("firms_"),
    "calendar": lambda c: c in ("day_of_week", "month", "is_weekend", "is_holiday", "is_ramadan"),
}

def group_of(c):
    for g, f in GROUPS.items():
        if f(c):
            return g
    raise KeyError(c)

def load(df=None):
    """Return dict with daily frame, feature list, X/y and the anchored split."""
    if df is None:
        df = pd.read_parquet(CFG["data"]["input_path"])
    feat, features = build_daily(df, CFG)
    X, y = model_matrix(feat, features)
    tr, va, te = split_masks(X.index, CFG)
    paired = feat[feat["pair_valid"]].dropna(subset=features + ["y_next"])
    hs = holdout_start(X.index, CFG)
    base = baselines(paired[paired.index < hs], paired[paired.index >= hs])
    return dict(df=df, feat=feat, features=features, X=X, y=y, tr=tr, te=te,
                X_tr=X[tr], y_tr=y[tr], X_te=X[te], y_te=y[te], hs=hs, base=base)

def fit_xgb(X_tr, y_tr, X_te):
    """Frozen-hyperparameter XGBoost on a train-only-scaled matrix."""
    sc = fit_scaler(X_tr)
    m = XGBRegressor(**HP["tuned"], **HP["fixed"]).fit(apply_scaler(sc, X_tr), np.asarray(y_tr))
    return m, m.predict(apply_scaler(sc, X_te)), sc

def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))

def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))

def dm_test(y, p_a, p_b, nw_lag=0, loss="se"):
    """Diebold-Mariano, a vs b. d = L(a) - L(b); negative => a better.
    nw_lag=0 reproduces the manuscript test (gamma0 only, HLN h=1, t(n-1)).
    nw_lag>0 uses a Bartlett (Newey-West) long-run variance with that many lags."""
    from scipy import stats
    y, a, b = (np.asarray(v, float) for v in (y, p_a, p_b))
    d = (a - y) ** 2 - (b - y) ** 2 if loss == "se" else np.abs(a - y) - np.abs(b - y)
    n = len(d); dbar = d.mean(); dc = d - dbar
    s = np.mean(dc * dc)
    for k in range(1, nw_lag + 1):
        s += 2 * (1 - k / (nw_lag + 1)) * np.mean(dc[k:] * dc[:-k])
    stat = dbar / np.sqrt(s / n) * np.sqrt((n - 1) / n)      # HLN factor for h=1
    p2 = float(2 * stats.t.cdf(-abs(stat), df=n - 1))
    p1 = float(stats.t.cdf(stat, df=n - 1))                  # H1: a better than b
    return {"dm_stat": float(stat), "p_two_sided": p2, "p_one_sided_a_better": p1,
            "mean_loss_diff": float(dbar), "n": int(n)}

def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out = {}; run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); out[k] = run
    return out

def dump(name, obj):
    json.dump(obj, open(OUT / name, "w"), indent=2, default=str)
