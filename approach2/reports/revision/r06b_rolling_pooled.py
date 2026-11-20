"""R06b - per-fold and pooled tests from the cached fold predictions (run r06 first).
Checks the per-fold XGBoost numbers against the frozen reports/rolling_backtest.csv."""
import glob, numpy as np, pandas as pd
from _common import *
from src.evaluate.metrics import paired_bootstrap_diff

P = pd.concat([pd.read_csv(f, index_col=0, parse_dates=True) for f in sorted(glob.glob(str(OUT / "r06_fold_*_predictions.csv")))])
M = pd.DataFrame([json.load(open(f)) for f in sorted(glob.glob(str(OUT / "r06_fold_*_meta.json")))])
M.to_csv(OUT / "r06_fold_dates.csv", index=False); P.to_csv(OUT / "r06_rolling_predictions.csv")
old = pd.read_csv("reports/rolling_backtest.csv"); old = old[old.model == "xgboost"].set_index("test_year")
MODELS = ["xgboost", "random_forest", "ridge", "lasso", "ols_pm_lags"]
rows = []
for yr, g in P.groupby("test_year"):
    r = {"test_year": yr, "n": len(g), "persistence_rmse": rmse(g.y, g.persistence)}
    for m in MODELS:
        r[f"{m}_rmse"] = rmse(g.y, g[m]); r[f"{m}_dm_p_vs_persist"] = dm_test(g.y, g[m], g.persistence)["p_two_sided"]
    b = paired_bootstrap_diff(g.y.values, g.xgboost.values, g.persistence.values, 1000, SEED)
    r.update({"xgb_d_rmse": b["delta_rmse"], "xgb_d_rmse_lo": b["delta_rmse_ci_lo"], "xgb_d_rmse_hi": b["delta_rmse_ci_hi"]})
    r["xgb_rmse_frozen_csv"] = old.loc[yr, "rmse"]; r["xgb_dm_p_frozen_csv"] = old.loc[yr, "dm_pvalue"]
    r["xgb_vs_ridge_dm_p"] = dm_test(g.y, g.xgboost, g.ridge)["p_two_sided"]
    r["xgb_vs_ols_dm_p"] = dm_test(g.y, g.xgboost, g.ols_pm_lags)["p_two_sided"]
    rows.append(r)
F = pd.DataFrame(rows)
for m in MODELS:
    h = holm(dict(zip(F.test_year, F[f"{m}_dm_p_vs_persist"]))); F[f"{m}_holm_p_6folds"] = F.test_year.map(h)
pre = F[F.test_year <= 2023]
h4 = holm(dict(zip(pre.test_year, pre["xgboost_dm_p_vs_persist"]))); F["xgboost_holm_p_4_preholdout_folds"] = F.test_year.map(h4)
F.to_csv(OUT / "r06_per_fold.csv", index=False)

def pooled(g, label):
    out = {"subset": label, "n": len(g), "persistence_rmse": rmse(g.y, g.persistence)}
    for m in MODELS:
        d0 = dm_test(g.y, g[m], g.persistence); d7 = dm_test(g.y, g[m], g.persistence, nw_lag=7)
        out[f"{m}_rmse"] = rmse(g.y, g[m]); out[f"{m}_dm_p"] = d0["p_two_sided"]; out[f"{m}_dm_p_nw7"] = d7["p_two_sided"]
        out[f"{m}_d_mae"] = float(np.mean(np.abs(g[m] - g.y)) - np.mean(np.abs(g.persistence - g.y)))
        out[f"{m}_dm_p_abs_loss"] = dm_test(g.y, g[m], g.persistence, loss="ae")["p_two_sided"]
    for m in ["ridge", "lasso", "ols_pm_lags", "random_forest"]:
        out[f"xgb_vs_{m}_dm_p"] = dm_test(g.y, g.xgboost, g[m])["p_two_sided"]
    return out
S = pd.DataFrame([pooled(P, "all 6 folds (2020-2025)"),
                  pooled(P[P.test_year <= 2023], "pre-holdout folds (test 2020-2023)"),
                  pooled(P[~P.in_holdout], "all fold days outside the holdout window"),
                  pooled(P[P.test_year >= 2024], "folds 2024-2025 (overlap holdout)")])
S.to_csv(OUT / "r06_pooled.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 60)
print(M[["test_year","train_first_day_t","train_last_day_t","n_train","test_first_day_t","test_last_day_t","n_test","n_test_in_holdout","n_train_in_holdout","xgb_params"]].to_string(index=False))
print(F.round(3).T.to_string()); print(S.round(3).T.to_string())
