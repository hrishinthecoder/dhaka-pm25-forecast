"""R03 - extra benchmarks on the same 219-day holdout, same protocol.
Also: RF / LightGBM re-run with the frozen protocol, R2 definition check."""
import numpy as np, pandas as pd
from _common import *
from _bench import linear_benchmarks, naive_benchmarks, arima_log
from src.models.train import train_models
from src.evaluate.metrics import reg_metrics, bootstrap_ci

D = load(); y = D["y_te"].values
sc = fit_scaler(D["X_tr"]); a, b = apply_scaler(sc, D["X_tr"]), apply_scaler(sc, D["X_te"])
models, tuning = train_models(a, D["y_tr"].values, a, D["y_tr"].values, CFG)   # exact main protocol
P = {k: m.predict(b) for k, m in models.items()}
P["persistence"] = D["base"]["persistence"]; P["climatology_doy"] = D["base"]["climatology"]
lin, info = linear_benchmarks(D["X_tr"], D["y_tr"], D["X_te"]); P.update(lin)
P.update(naive_benchmarks(D["feat"], D["hs"], D["X_te"].index))
P["arima201_log"], info["arima201_log"] = arima_log(D["feat"], D["hs"], D["X_te"].index)
pd.DataFrame(P, index=D["X_te"].index).assign(y=y).to_csv(OUT / "r03_holdout_predictions.csv")
rows = []
for k, p in P.items():
    ok = ~np.isnan(np.asarray(p, float)); yy, pp = y[ok], np.asarray(p, float)[ok]
    r = {"model": k, **reg_metrics(yy, pp), "r2_1_minus_sse_sst": r2(yy, pp),
         "squared_pearson_r": float(np.corrcoef(yy, pp)[0, 1] ** 2)}
    ci = bootstrap_ci(yy, pp, 1000, SEED); r.update({"rmse_ci_lo": ci["rmse_ci_lo"], "rmse_ci_hi": ci["rmse_ci_hi"],
                                                    "r2_ci_lo": ci["r2_ci_lo"], "r2_ci_hi": ci["r2_ci_hi"]})
    if k != "persistence":
        d = dm_test(yy, pp, P["persistence"][ok]); r["d_rmse_vs_persist"] = r["rmse"] - rmse(yy, P["persistence"][ok]); r["dm_p_vs_persist"] = d["p_two_sided"]
    if k != "xgboost":
        d = dm_test(yy, P["xgboost"][ok], pp); r["xgb_minus_model_rmse"] = rmse(yy, P["xgboost"][ok]) - r["rmse"]; r["dm_p_xgb_vs_model"] = d["p_two_sided"]
    rows.append(r)
T = pd.DataFrame(rows).sort_values("rmse"); T.to_csv(OUT / "r03_benchmarks.csv", index=False)
dump("r03_benchmark_info.json", {"tuning": tuning, "linear": info})
pd.set_option("display.width", 250); print(T.round(3).to_string(index=False)); print(json.dumps(info, indent=1)[:1800])
