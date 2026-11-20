"""R05 - significance details on the 219-day holdout: DM specification, one/two-sided,
Newey-West variants, Holm correction, loss-differential autocorrelation, day-level vs
moving-block bootstrap intervals for dRMSE and dMAE, R2 definition."""
import numpy as np, pandas as pd
from _common import *
from src.evaluate.metrics import paired_bootstrap_diff

P = pd.read_csv(OUT / "r03_holdout_predictions.csv", index_col=0, parse_dates=True)
frozen = pd.read_csv("reports/significance_vs_persistence.csv").set_index("model")
y, per = P.y.values, P.persistence.values
o = {"dm_spec": "loss differential d_t = (yhat_model - y)^2 - (yhat_persistence - y)^2; H0: E[d]=0; "
                "variance = gamma_0 / n (h=1, no autocovariance terms); Harvey-Leybourne-Newbold factor sqrt((n-1)/n); "
                "reference distribution Student t with n-1 df; two-sided p reported in the submitted manuscript"}
d = (P.xgboost.values - y) ** 2 - (per - y) ** 2; dc = d - d.mean()
o["loss_diff_acf"] = {f"lag{k}": float(np.sum(dc[k:] * dc[:-k]) / np.sum(dc * dc)) for k in (1, 2, 3, 7)}
gaps = np.diff(P.index.values).astype("timedelta64[D]").astype(int)
o["holdout_consecutive_day_steps"] = int((gaps == 1).sum()); o["holdout_steps_with_gap"] = int((gaps > 1).sum()); o["largest_gap_days"] = int(gaps.max())
o["xgboost"] = {f"nw_lag_{L}": dm_test(y, P.xgboost.values, per, nw_lag=L) for L in (0, 1, 3, 7, 14)}
o["xgboost_abs_loss"] = dm_test(y, P.xgboost.values, per, loss="ae")
# Holm across the three ML models. RF/LightGBM p-values: frozen report (library versions for those two are unpinned)
p_frozen = {m: float(frozen.loc[m, "dm_pvalue"]) for m in ("random_forest", "xgboost", "lightgbm")}
p_rerun = {m: dm_test(y, P[m].values, per)["p_two_sided"] for m in ("random_forest", "xgboost", "lightgbm")}
o["holm_3_ml_models_frozen_p"] = {"raw": p_frozen, "holm": holm(p_frozen)}
o["holm_3_ml_models_rerun_p"] = {"raw": p_rerun, "holm": holm(p_rerun)}
allm = [c for c in P.columns if c not in ("y", "persistence", "weekly_seasonal_persistence")]
pa = {m: dm_test(y, P[m].values, per)["p_two_sided"] for m in allm}
o["holm_all_12_models_vs_persistence"] = {"raw": pa, "holm": holm(pa)}

def mbb(a, b, block, n_boot=2000, seed=SEED):
    """Moving-block bootstrap (overlapping blocks, holdout rows in time order) for dRMSE, dMAE."""
    rng = np.random.default_rng(seed); n = len(y); k = int(np.ceil(n / block)); starts = np.arange(n - block + 1)
    dr, dm = [], []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, k)])[:n]
        ea, eb = a[idx] - y[idx], b[idx] - y[idx]
        dr.append(np.sqrt(np.mean(ea ** 2)) - np.sqrt(np.mean(eb ** 2))); dm.append(np.mean(np.abs(ea)) - np.mean(np.abs(eb)))
    return {"d_rmse_ci": [float(v) for v in np.percentile(dr, [2.5, 97.5])], "d_mae_ci": [float(v) for v in np.percentile(dm, [2.5, 97.5])]}
rows = []
for m in ("xgboost", "random_forest", "lightgbm", "lasso", "ridge", "ols_pm_lags"):
    a = P[m].values; iid = paired_bootstrap_diff(y, a, per, 1000, SEED)
    r = {"model": m, "d_rmse": iid["delta_rmse"], "d_mae": iid["delta_mae"],
         "iid_d_rmse_ci": [iid["delta_rmse_ci_lo"], iid["delta_rmse_ci_hi"]], "iid_d_mae_ci": [iid["delta_mae_ci_lo"], iid["delta_mae_ci_hi"]]}
    for L in (7, 14, 21):
        b = mbb(a, per, L); r[f"mbb{L}_d_rmse_ci"] = b["d_rmse_ci"]; r[f"mbb{L}_d_mae_ci"] = b["d_mae_ci"]
    rows.append(r)
pd.DataFrame(rows).to_json(OUT / "r05_bootstrap_intervals.json", orient="records", indent=1)
o["r2_definition"] = {"r2_score_sklearn": r2(y, P.xgboost.values), "squared_pearson": float(np.corrcoef(y, P.xgboost.values)[0, 1] ** 2),
                      "note": "all models scored with sklearn.metrics.r2_score = 1 - SSE/SST (src/evaluate/metrics.py)"}
dump("r05_significance.json", o); print(json.dumps(o, indent=1)[:4200])
for r in rows: print({k: (np.round(v, 2).tolist() if isinstance(v, list) else round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
