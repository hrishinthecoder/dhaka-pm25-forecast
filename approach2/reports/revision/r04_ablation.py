"""R04 - feature-group ablation with frozen XGBoost hyperparameters (xgboost 3.2.0),
plus pairwise feature correlation. Same train/holdout rows for every variant."""
import itertools, numpy as np, pandas as pd
from _common import *
from src.evaluate.metrics import reg_metrics

D = load(); f = D["features"]; y = D["y_te"].values
g = {k: [c for c in f if group_of(c) == k] for k in GROUPS}
sets = {"full_45": f, "pm_history_only": g["pm_history"], "pm_history+calendar": g["pm_history"] + g["calendar"],
        "pm_history+era5": g["pm_history"] + g["era5"], "pm_history+firms": g["pm_history"] + g["firms"],
        "no_era5": [c for c in f if c not in g["era5"]], "no_firms": [c for c in f if c not in g["firms"]],
        "no_calendar": [c for c in f if c not in g["calendar"]], "no_pm_history": [c for c in f if c not in g["pm_history"]],
        "era5_only": g["era5"], "calendar_only": g["calendar"], "era5+calendar": g["era5"] + g["calendar"]}
P = {k: fit_xgb(D["X_tr"][c], D["y_tr"], D["X_te"][c])[1] for k, c in sets.items()}
per = D["base"]["persistence"]; rows = []
for k, p in P.items():
    r = {"variant": k, "n_features": len(sets[k]), **reg_metrics(y, p)}
    r["dm_p_vs_persistence"] = dm_test(y, p, per)["p_two_sided"]
    if k != "full_45":
        d = dm_test(y, p, P["full_45"]); r["rmse_minus_full"] = r["rmse"] - rmse(y, P["full_45"]); r["dm_p_vs_full"] = d["p_two_sided"]
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(OUT / "r04_ablation.csv", index=False)
pd.DataFrame(P, index=D["X_te"].index).assign(y=y).to_csv(OUT / "r04_ablation_predictions.csv")
C = D["X_tr"].corr().abs(); pairs = [(a, b, C.loc[a, b]) for a, b in itertools.combinations(f, 2)]
hi = sorted([p for p in pairs if p[2] > 0.9], key=lambda t: -t[2])
pd.DataFrame(hi, columns=["a", "b", "abs_r"]).to_csv(OUT / "r04_high_corr_pairs.csv", index=False)
cross = [(a, b, round(r, 3)) for a, b, r in pairs if r > 0.7 and group_of(a) != group_of(b)]
o = {"group_sizes": {k: len(v) for k, v in g.items()}, "n_pairs": len(pairs), "n_pairs_abs_r_gt_0.9": len(hi),
     "n_pairs_gt_0.9_within_group": sum(group_of(a) == group_of(b) for a, b, _ in hi),
     "cross_group_pairs_abs_r_gt_0.7": cross}
dump("r04_groups_corr.json", o); pd.set_option("display.width", 250); print(T.round(3).to_string(index=False)); print(o)
