"""Regression / classification metrics, bootstrap CIs, seasonal split, ACF.

The test window is thin (~230 pairs), so bootstrap CIs are MANDATORY for every
regression metric (PROMPT_2). Seasons follow the South-Asian monsoon calendar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)

# Monsoon-calendar seasons (covers all 12 months).
WINTER_DRY = {12, 1, 2}        # DJF dry/winter (peak PM2.5)
MONSOON = {6, 7, 8, 9}         # JJAS wet/monsoon (washout)
# everything else (Mar-May, Oct-Nov) = transition (pre/post-monsoon)


def season_of(month: int) -> str:
    if month in WINTER_DRY:
        return "winter_dry"
    if month in MONSOON:
        return "monsoon"
    return "transition"


def _clean(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    m = ~(np.isnan(y) | np.isnan(yhat))
    return y[m], yhat[m]


def reg_metrics(y, yhat) -> dict:
    y, yhat = _clean(y, yhat)
    if len(y) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "bias": np.nan, "n": 0}
    return {"rmse": float(np.sqrt(mean_squared_error(y, yhat))),
            "mae": float(mean_absolute_error(y, yhat)),
            "r2": float(r2_score(y, yhat)),
            "bias": float(np.mean(yhat - y)),
            "n": int(len(y))}


def bootstrap_ci(y, yhat, n_boot=1000, seed=42, alpha=0.05) -> dict:
    """Percentile bootstrap 95% CI for rmse, mae, r2, bias."""
    y, yhat = _clean(y, yhat)
    n = len(y)
    rng = np.random.default_rng(seed)
    acc = {"rmse": [], "mae": [], "r2": [], "bias": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], yhat[idx]
        acc["rmse"].append(np.sqrt(mean_squared_error(yb, pb)))
        acc["mae"].append(mean_absolute_error(yb, pb))
        acc["r2"].append(r2_score(yb, pb))
        acc["bias"].append(np.mean(pb - yb))
    out = {}
    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for k, v in acc.items():
        lo, hi = np.percentile(v, [lo_p, hi_p])
        out[f"{k}_ci_lo"] = float(lo)
        out[f"{k}_ci_hi"] = float(hi)
    return out


def seasonal_metrics(dates: pd.DatetimeIndex, y, yhat) -> pd.DataFrame:
    """reg_metrics per monsoon season (season assigned by day-t+1 month)."""
    seasons = pd.Series(pd.DatetimeIndex(dates).month, index=range(len(dates))).map(season_of)
    y = pd.Series(np.asarray(y, float))
    yhat = pd.Series(np.asarray(yhat, float))
    rows = []
    for s in ["winter_dry", "transition", "monsoon"]:
        m = (seasons == s).values
        r = reg_metrics(y[m], yhat[m])
        r["season"] = s
        rows.append(r)
    return pd.DataFrame(rows).set_index("season")


def residual_acf(resid, nlags=30) -> np.ndarray:
    """Autocorrelation of residuals (numpy; checks temporal independence)."""
    x = np.asarray(resid, float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return np.zeros(nlags + 1)
    out = [1.0]
    for k in range(1, nlags + 1):
        out.append(float(np.sum(x[k:] * x[:-k]) / denom))
    return np.array(out)


def _paired_errors(y, yhat_model, yhat_base):
    """Aligned absolute & squared errors for model vs baseline on the SAME days."""
    y = np.asarray(y, float)
    a = np.asarray(yhat_model, float)
    b = np.asarray(yhat_base, float)
    m = ~(np.isnan(y) | np.isnan(a) | np.isnan(b))
    y, a, b = y[m], a[m], b[m]
    ae_m, ae_b = np.abs(a - y), np.abs(b - y)
    se_m, se_b = (a - y) ** 2, (b - y) ** 2
    return ae_m, ae_b, se_m, se_b


def _rmse_from_se(se):
    return float(np.sqrt(np.mean(se)))


def paired_bootstrap_diff(y, yhat_model, yhat_base, n_boot=1000, seed=42, alpha=0.05) -> dict:
    """Paired bootstrap of error differences (model − baseline), same days.

    Returns ΔMAE and ΔRMSE point estimates + 95% CIs. Negative ⇒ model better.
    The CI excluding 0 is the test that the model beats the baseline.
    For ΔRMSE we resample the paired squared errors and recompute BOTH RMSEs per
    resample, then difference (RMSE is not a mean, so we cannot difference per-day).
    """
    ae_m, ae_b, se_m, se_b = _paired_errors(y, yhat_model, yhat_base)
    n = len(ae_m)
    delta_mae = float(np.mean(ae_m) - np.mean(ae_b))
    delta_rmse = _rmse_from_se(se_m) - _rmse_from_se(se_b)
    rng = np.random.default_rng(seed)
    bmae, brmse = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bmae.append(np.mean(ae_m[idx]) - np.mean(ae_b[idx]))
        brmse.append(_rmse_from_se(se_m[idx]) - _rmse_from_se(se_b[idx]))
    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)
    mae_lo, mae_hi = np.percentile(bmae, [lo_p, hi_p])
    rmse_lo, rmse_hi = np.percentile(brmse, [lo_p, hi_p])
    return {"delta_mae": delta_mae,
            "delta_mae_ci_lo": float(mae_lo), "delta_mae_ci_hi": float(mae_hi),
            "delta_rmse": float(delta_rmse),
            "delta_rmse_ci_lo": float(rmse_lo), "delta_rmse_ci_hi": float(rmse_hi)}


def diebold_mariano(y, yhat_model, yhat_base, h=1) -> dict:
    """Diebold–Mariano test, squared-error loss, with HLN small-sample correction.

    Loss differential d_i = se_model_i − se_base_i. HAC (Newey–West) long-run
    variance with bandwidth h−1 (h=1 ⇒ γ0 only). Harvey–Leybourne–Newbold (1997)
    small-sample correction; statistic referred to t with n−1 df.
    d̄ < 0 (DM < 0) ⇒ model has lower squared error than the baseline.
    """
    from scipy import stats
    _, _, se_m, se_b = _paired_errors(y, yhat_model, yhat_base)
    d = se_m - se_b
    n = len(d)
    dbar = float(np.mean(d))
    dc = d - dbar
    gamma0 = float(np.mean(dc * dc))
    s = gamma0
    for k in range(1, h):
        gk = float(np.mean(dc[k:] * dc[:-k]))
        s += 2 * gk
    var_dbar = s / n
    if var_dbar <= 0:
        return {"dm_stat": float("nan"), "dm_pvalue": float("nan")}
    dm = dbar / np.sqrt(var_dbar)
    k_corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_star = float(dm * k_corr)
    pval = float(2 * stats.t.cdf(-abs(dm_star), df=n - 1))
    return {"dm_stat": dm_star, "dm_pvalue": pval}


def significance_vs_persistence(y, preds: dict, baseline="persistence",
                                models=("random_forest", "xgboost", "lightgbm",
                                        "climatology"),
                                n_boot=1000, seed=42, h=1) -> pd.DataFrame:
    """Paired bootstrap (ΔMAE, ΔRMSE) + Diebold–Mariano for each model vs the
    persistence baseline. `climatology` is included as a sanity row (it should NOT
    beat persistence). Returns one row per available model."""
    if baseline not in preds:
        raise ValueError(f"baseline '{baseline}' not in preds: {list(preds)}")
    base = preds[baseline]
    rows = []
    for name in models:
        if name not in preds or name == baseline:
            continue
        boot = paired_bootstrap_diff(y, preds[name], base, n_boot=n_boot, seed=seed)
        dm = diebold_mariano(y, preds[name], base, h=h)
        rows.append({"model": name, **boot, **dm})
    cols = ["model", "delta_mae", "delta_mae_ci_lo", "delta_mae_ci_hi",
            "delta_rmse", "delta_rmse_ci_lo", "delta_rmse_ci_hi",
            "dm_stat", "dm_pvalue"]
    return pd.DataFrame(rows, columns=cols)


def classification_metrics(y_true_cat, y_pred_cat, labels) -> dict:
    """Accuracy + macro-F1 + confusion matrix for AQI categories."""
    yt = pd.Series(y_true_cat).astype(str)
    yp = pd.Series(y_pred_cat).astype(str)
    m = yt.notna() & yp.notna() & (yt != "nan") & (yp != "nan")
    yt, yp = yt[m], yp[m]
    cm = confusion_matrix(yt, yp, labels=labels)
    return {"accuracy": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, labels=labels,
                                       average="macro", zero_division=0)),
            "confusion_matrix": cm,
            "n": int(len(yt))}
