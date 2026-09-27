"""
models_tree.py — gradient-boosting models (the headline) with leakage-safe tuning.

WHAT THIS MODULE GUARANTEES
    Every learned transform is fit on TRAIN ROWS ONLY, inside each walk-forward fold:
        - median imputation of residual feature NaNs (fit on the fold's train),
        - an IsolationForest anomaly FLAG appended as a feature (fit on the fold's train,
          contamination from config) — rows are never deleted, only flagged,
        - early stopping uses an INNER time-ordered validation slice taken from the train
          portion (never the test fold), so the test fold stays unseen during fitting.
    Optuna optimises the walk-forward CV RMSE; the final 12-month hold-out is never passed
    to Optuna or to any fold here.

MODELS
    LightGBM and XGBoost are the primary models; RandomForest is a secondary tree baseline.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor

from validation import make_walk_forward

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ----------------------------------------------------------------------------
# Fold-internal preprocessing (fit on train only)
# ----------------------------------------------------------------------------
def _fit_medians(X: pd.DataFrame) -> pd.Series:
    """Column medians from TRAIN rows only — used to fill residual feature NaNs."""
    return X.median(numeric_only=True)


def _apply_medians(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """Fill NaNs with the train medians; any column still all-NaN becomes 0."""
    return X.fillna(medians).fillna(0.0)


def _anomaly_flag(X_train_imp, X_apply_imp, cfg) -> np.ndarray:
    """Train an IsolationForest on TRAIN features, return a 0/1 anomaly flag for X_apply.

    The flag (1 == flagged anomalous) is added as a feature so the model can down-weight
    odd days; we never drop rows. Fit on train only -> no leakage.
    """
    iso_cfg = cfg["preprocessing"]["outliers"]["isolation_forest"]
    if not iso_cfg.get("enabled", False):
        return np.zeros(len(X_apply_imp), dtype=int)
    iso = IsolationForest(contamination=iso_cfg["contamination"],
                          random_state=cfg["project"]["random_seed"])
    iso.fit(X_train_imp)
    pred = iso.predict(X_apply_imp)  # -1 anomaly, 1 normal
    return (pred == -1).astype(int)


def _inner_time_split(n: int, val_frac: float = 0.15, min_val: int = 30):
    """Split a fold's train rows into (inner_train, inner_val) by TIME order.

    The last `val_frac` of the rows (most recent) become the early-stopping validation
    set. Keeping it time-ordered (not random) means early stopping is judged on the future
    relative to inner-train — consistent with the forecasting setup and leakage-free w.r.t.
    the outer test fold.
    """
    n_val = max(min_val, int(round(n * val_frac)))
    n_val = min(n_val, n - min_val) if n - min_val > 0 else n_val
    cut = n - n_val
    return np.arange(cut), np.arange(cut, n)


def _make_model(name: str, params: dict, cfg: dict):
    """Instantiate a model with the given params and the global seed."""
    seed = cfg["project"]["random_seed"]
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbosity=-1, **params)
    if name == "xgboost":
        import xgboost as xgb
        es = cfg["models"]["xgboost"].get("early_stopping_rounds", 50)
        return xgb.XGBRegressor(random_state=seed, n_jobs=-1, eval_metric="rmse",
                                early_stopping_rounds=es, **params)
    if name == "random_forest":
        rf = cfg["models"]["random_forest"]
        return RandomForestRegressor(
            n_estimators=rf["n_estimators"], max_depth=rf["max_depth"],
            min_samples_leaf=rf["min_samples_leaf"], n_jobs=-1, random_state=seed)
    raise ValueError(name)


def fit_predict_fold(name, params, X_tr, y_tr, X_te, cfg, use_early_stop=True):
    """Train one model on a fold's train rows and predict its test rows (leakage-safe).

    Pipeline order: median-impute (fit train) -> append IsolationForest flag (fit train)
    -> fit model (early stopping on an inner time-split of the train) -> predict test.
    """
    medians = _fit_medians(X_tr)
    Xtr_imp = _apply_medians(X_tr, medians)
    Xte_imp = _apply_medians(X_te, medians)

    # Anomaly flag as an extra feature (fit IForest on train-imputed features).
    flag_tr = _anomaly_flag(Xtr_imp, Xtr_imp, cfg)
    flag_te = _anomaly_flag(Xtr_imp, Xte_imp, cfg)
    Xtr_imp = Xtr_imp.assign(anomaly_flag=flag_tr)
    Xte_imp = Xte_imp.assign(anomaly_flag=flag_te)

    model = _make_model(name, params, cfg)

    if name == "lightgbm" and use_early_stop:
        import lightgbm as lgb
        itr, ival = _inner_time_split(len(Xtr_imp))
        rounds = cfg["models"]["lightgbm"].get("early_stopping_rounds", 50)
        model.fit(Xtr_imp.iloc[itr], y_tr[itr],
                  eval_set=[(Xtr_imp.iloc[ival], y_tr[ival])],
                  callbacks=[lgb.early_stopping(rounds, verbose=False)])
    elif name == "xgboost" and use_early_stop:
        itr, ival = _inner_time_split(len(Xtr_imp))
        model.fit(Xtr_imp.iloc[itr], y_tr[itr],
                  eval_set=[(Xtr_imp.iloc[ival], y_tr[ival])], verbose=False)
    else:
        model.fit(Xtr_imp, y_tr)

    return model.predict(Xte_imp)


# ----------------------------------------------------------------------------
# Optuna tuning on the walk-forward CV score (RMSE)
# ----------------------------------------------------------------------------
def _suggest(trial, name, space):
    """Translate a config search-space dict into Optuna suggestions."""
    p = {}
    log_params = {"learning_rate", "reg_alpha", "reg_lambda"}
    int_params = {"n_estimators", "num_leaves", "max_depth", "min_child_samples",
                  "min_child_weight"}
    for key, (lo, hi) in space.items():
        if key in int_params:
            p[key] = trial.suggest_int(key, int(lo), int(hi))
        elif key in log_params:
            p[key] = trial.suggest_float(key, lo, hi, log=True)
        else:
            p[key] = trial.suggest_float(key, lo, hi)
    return p


def tune(name, X, y, cfg):
    """Run Optuna to minimise mean walk-forward CV RMSE for a model.

    Returns (best_params, study). The hold-out is NOT part of X here — only dev rows are
    passed in, so Optuna never sees the final test data.
    """
    from evaluate import rmse
    space = cfg["models"][name]["search_space"]
    n_trials = cfg["models"][name]["optuna"]["n_trials"]
    tscv = make_walk_forward(cfg, len(X))
    y = np.asarray(y, float)

    def objective(trial):
        """One Optuna trial: build params, return the mean walk-forward CV RMSE (the value
        Optuna minimises). Only dev folds are used, so the hold-out never informs tuning."""
        params = _suggest(trial, name, space)
        scores = []
        for tr_idx, te_idx in tscv.split(X):
            preds = fit_predict_fold(name, params, X.iloc[tr_idx], y[tr_idx],
                                     X.iloc[te_idx], cfg)
            scores.append(rmse(y[te_idx], preds))
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=cfg["project"]["random_seed"])
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study


def fit_full(name, params, X, y, cfg):
    """Fit a model on the FULL dev set and return a self-contained prediction bundle.

    The bundle stores the train medians and the fitted IsolationForest so the exact same
    leakage-safe preprocessing is reapplied at prediction time (e.g. on the hold-out). The
    hold-out is never passed here — only dev rows — so nothing about it informs the fit.
    """
    y = np.asarray(y, float)
    medians = _fit_medians(X)
    X_imp = _apply_medians(X, medians)

    iso = None
    iso_cfg = cfg["preprocessing"]["outliers"]["isolation_forest"]
    if iso_cfg.get("enabled", False):
        iso = IsolationForest(contamination=iso_cfg["contamination"],
                              random_state=cfg["project"]["random_seed"]).fit(X_imp)
        X_imp = X_imp.assign(anomaly_flag=(iso.predict(X_imp) == -1).astype(int))
    else:
        X_imp = X_imp.assign(anomaly_flag=0)

    model = _make_model(name, params, cfg)
    if name in ("lightgbm", "xgboost"):
        itr, ival = _inner_time_split(len(X_imp))
        if name == "lightgbm":
            import lightgbm as lgb
            rounds = cfg["models"]["lightgbm"].get("early_stopping_rounds", 50)
            model.fit(X_imp.iloc[itr], y[itr], eval_set=[(X_imp.iloc[ival], y[ival])],
                      callbacks=[lgb.early_stopping(rounds, verbose=False)])
        else:
            model.fit(X_imp.iloc[itr], y[itr],
                      eval_set=[(X_imp.iloc[ival], y[ival])], verbose=False)
    else:
        model.fit(X_imp, y)
    return {"name": name, "model": model, "medians": medians, "iso": iso,
            "feature_cols": list(X.columns)}


def predict_bundle(bundle, X) -> np.ndarray:
    """Apply a fitted bundle to new rows (same impute + anomaly-flag preprocessing)."""
    X = X[bundle["feature_cols"]]
    X_imp = _apply_medians(X, bundle["medians"])
    if bundle["iso"] is not None:
        X_imp = X_imp.assign(anomaly_flag=(bundle["iso"].predict(X_imp) == -1).astype(int))
    else:
        X_imp = X_imp.assign(anomaly_flag=0)
    return bundle["model"].predict(X_imp)


def cv_fold_metrics(name, params, X, y, cfg):
    """Walk-forward per-fold metrics for a fixed param set (used for the results table)."""
    from evaluate import all_metrics
    tscv = make_walk_forward(cfg, len(X))
    y = np.asarray(y, float)
    rows = []
    for i, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        preds = fit_predict_fold(name, params, X.iloc[tr_idx], y[tr_idx],
                                 X.iloc[te_idx], cfg)
        m = all_metrics(y[te_idx], preds)
        m["fold"] = i
        m["n_test"] = len(te_idx)
        rows.append(m)
    return pd.DataFrame(rows)
