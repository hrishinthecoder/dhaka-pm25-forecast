"""R06a - rolling-origin folds: exact dates and per-day predictions.

Protocol identical to src/evaluate/rolling_backtest.py: train day-t <= 31 Dec y,
test day-t in year y+1, scaler fit on the fold's train rows, XGBoost grid re-tuned
by TimeSeriesSplit(5) inside the fold's training window (src.models.train._tune),
random forest fixed (500 trees, seed 42). Adds ridge, lasso and lag-only OLS.
Each (fold, model) result is cached so the script can be resumed:
    python reports/revision/r06_rolling_folds.py [--fresh] [train_end_year ...]
--fresh ignores the cached fold files and recomputes them.
"""
import sys, time, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from _common import *
from _bench import linear_benchmarks
from src.models.train import _tune

D = load(); X, y = D["X"], D["y"]
GRID = {"max_depth": [3, 5, 7], "learning_rate": [0.03, 0.1], "n_estimators": [300, 600]}   # as in src/models/train.py
FRESH = "--fresh" in sys.argv
YEARS = [int(a) for a in sys.argv[1:] if a != "--fresh"] or [2019, 2020, 2021, 2022, 2023, 2024]
for yr in YEARS:
    end = pd.Timestamp(f"{yr}-12-31"); tr = X.index <= end; te = X.index.year == yr + 1
    X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
    sc = fit_scaler(X_tr); a, b = apply_scaler(sc, X_tr), apply_scaler(sc, X_te)
    f = OUT / f"r06_fold_{yr + 1}_predictions.csv"
    fr = pd.read_csv(f, index_col=0, parse_dates=True) if f.exists() and not FRESH else pd.DataFrame(index=X_te.index)
    mf = OUT / f"r06_fold_{yr + 1}_meta.json"; meta = json.load(open(mf)) if mf.exists() and not FRESH else {}
    fr["y"] = y_te.values; fr["persistence"] = X_te["pm25_t"].values; fr["test_year"] = yr + 1
    fr["in_holdout"] = fr.index >= D["hs"]
    t0 = time.time()
    if "xgboost" not in fr or "xgb_params" not in meta:
        best, params, cv = _tune(XGBRegressor(random_state=SEED, n_jobs=-1, tree_method="hist",
                                              objective="reg:squarederror"), GRID, a, y_tr.values)
        best.fit(a, y_tr.values); fr["xgboost"] = best.predict(b); meta["xgb_params"] = params; meta["xgb_cv_rmse"] = cv
    if "random_forest" not in fr:
        fr["random_forest"] = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1).fit(a, y_tr.values).predict(b)
    if "ridge" not in fr or "ridge_alpha" not in meta:
        lin, info = linear_benchmarks(X_tr, y_tr, X_te)
        for k in ("ridge", "lasso", "ols_pm_lags"):
            fr[k] = lin[k]
        meta["ridge_alpha"] = info["ridge"]["alpha"]; meta["lasso_alpha"] = info["lasso"]["alpha"]
    meta.update({"test_year": yr + 1, "train_first_day_t": X_tr.index.min().date(), "train_last_day_t": X_tr.index.max().date(),
                 "n_train": int(tr.sum()), "test_first_day_t": X_te.index.min().date(), "test_last_day_t": X_te.index.max().date(),
                 "n_test": int(te.sum()), "n_test_in_holdout": int(fr.in_holdout.sum()),
                 "n_train_in_holdout": int((X_tr.index >= D["hs"]).sum())})
    fr.to_csv(f); dump(mf.name, meta)
    print(f"fold {yr + 1} done in {time.time() - t0:.0f}s", flush=True)
