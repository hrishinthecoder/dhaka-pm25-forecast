"""R13 - revised manuscript figures (1200 dpi, sans-serif lettering of at least 8 pt), drawn only from files in reports/revision/."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import *

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 1200, "figure.dpi": 100})
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
K, G, B, R = "#222222", "#8a8a8a", "#1f5fa8", "#b5372c"
D = load(); feat = D["feat"]

# Fig 1: record and split
d = feat[feat.valid]; bad = pd.read_csv(OUT / "r01_fault985_days.csv", index_col=0, parse_dates=True); bad = bad[bad.valid]
fig, ax = plt.subplots(figsize=(7.0, 2.6))
ax.axvspan(D["X_tr"].index.min(), D["hs"], color="#e9eef5", lw=0); ax.axvspan(D["hs"], D["X_te"].index.max() + pd.Timedelta(days=1), color="#f6e3df", lw=0)
ax.plot(d.index, d.pm25_mean, ".", ms=1.6, color=K, rasterized=True)
ax.plot(bad.index, d.pm25_mean.reindex(bad.index), "o", ms=3.5, mfc="none", mec=R, mew=0.8, label="valid day containing 1-3 hours of the 985 fault value (n = 20)")
ax.set_ylabel("Daily mean PM$_{2.5}$ (µg m$^{-3}$)"); ax.set_xlim(pd.Timestamp("2016-10-01"), pd.Timestamp("2025-05-01")); ax.set_ylim(0, 440)
ax.text(pd.Timestamp("2020-06-01"), 414, "training window: 1,341 rows (day t 16 Nov 2016 - 22 Mar 2024)", ha="center", va="top", fontsize=8)
ax.text(pd.Timestamp("2024-09-20"), 414, "holdout:\n219 rows", ha="center", va="top", fontsize=8)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG / "fig1_record_split.png"); plt.close(fig)

# Fig 2: predicted vs observed
P = pd.read_csv(OUT / "r03_holdout_predictions.csv", index_col=0, parse_dates=True)
fig, ax = plt.subplots(figsize=(3.4, 3.4)); m = 400
ax.plot([0, m], [0, m], "-", color=G, lw=0.8); ax.axvline(240, color=R, lw=0.7, ls="--")
ax.plot(P.y, P.xgboost, "o", ms=2.6, mfc="none", mec=B, mew=0.6)
ax.text(246, 20, "observed > 240:\n11 of 11 days\nunderpredicted", fontsize=8, color=R, va="bottom")
ax.set_xlim(0, m); ax.set_ylim(0, m); ax.set_aspect("equal"); ax.set_xlabel("Observed next-day PM$_{2.5}$ (µg m$^{-3}$)"); ax.set_ylabel("XGBoost forecast (µg m$^{-3}$)")
fig.tight_layout(); fig.savefig(FIG / "fig2_pred_vs_obs.png"); plt.close(fig)

# Fig 3: rolling folds
F = pd.read_csv(OUT / "r06_per_fold.csv"); x = np.arange(len(F))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.2, 4.6), sharex=True)
for a in (a1, a2): a.axvspan(3.5, 5.5, color="#f6e3df", lw=0)
a1.plot(x, F.persistence_rmse, "s--", color=G, ms=4, lw=0.9, label="persistence"); a1.plot(x, F.xgboost_rmse, "o-", color=B, ms=4, lw=0.9, label="XGBoost")
a1.plot(x, F.ols_pm_lags_rmse, "^:", color=K, ms=4, lw=0.9, label="OLS on five PM$_{2.5}$ lags"); a1.set_ylabel("Test RMSE (µg m$^{-3}$)"); a1.legend(frameon=False, fontsize=8, loc="upper left")
a2.errorbar(x, F.xgb_d_rmse, yerr=[F.xgb_d_rmse - F.xgb_d_rmse_lo, F.xgb_d_rmse_hi - F.xgb_d_rmse], fmt="o", color=B, ms=4, capsize=3, lw=0.9)
a2.text(4.5, 8.6, "test days overlap the holdout", ha="center", va="top", fontsize=8, color=R)
a2.axhline(0, color=K, lw=0.7); a2.set_ylabel("ΔRMSE, XGBoost − persistence\n(µg m$^{-3}$, 95% interval)")
a2.set_xticks(x); a2.set_xticklabels([f"{y}\n(n = {n})" for y, n in zip(F.test_year, F.n)]); a2.set_xlabel("Test year of the fold")
fig.tight_layout(); fig.savefig(FIG / "fig3_rolling_folds.png"); plt.close(fig)

# Fig 4: SHAP share vs ablation
S = json.load(open(OUT / "r08_group_shap.json"))["holdout"]["group_share_pct"]; A = pd.read_csv(OUT / "r04_ablation.csv").set_index("variant")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1, 1.5]})
lab = {"era5": "ERA5 meteorology (25)", "pm_history": "PM$_{2.5}$ history (9)", "calendar": "calendar (5)", "firms": "FIRMS fire (6)"}; o = ["era5", "pm_history", "calendar", "firms"]
a1.barh([lab[k] for k in o][::-1], [S[k] for k in o][::-1], color=B, height=0.6)
for i, k in enumerate(o[::-1]): a1.text(S[k] + 1, i, f"{S[k]:.1f}%", va="center", fontsize=8)
a1.set_xlim(0, 72); a1.set_xlabel("Share of summed mean |SHAP| (%)"); a1.text(-0.02, 1.04, "a", transform=a1.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")
v = [("full_45", "all 45 features"), ("no_firms", "without FIRMS (39)"), ("no_era5", "without ERA5 (20)"), ("pm_history+calendar", "PM$_{2.5}$ history + calendar (14)"),
     ("pm_history_only", "PM$_{2.5}$ history only (9)"), ("era5_only", "ERA5 only (25)")]
a2.barh([n for _, n in v][::-1], [A.loc[k, "rmse"] for k, _ in v][::-1], color=[G if k != "full_45" else B for k, _ in v][::-1], height=0.6)
for i, (k, _) in enumerate(v[::-1]): a2.text(27.15 if k == "era5_only" else A.loc[k, "rmse"] + 0.15, i, f"{A.loc[k, 'rmse']:.2f}", va="center", fontsize=8, color="white" if k == "era5_only" else "black")
a2.axvline(rmse(P.y, P.persistence), color=R, lw=0.8, ls="--"); a2.text(rmse(P.y, P.persistence) + 0.12, 3.5, "persistence\n36.02", color=R, fontsize=8, ha="left", va="center")
a2.set_xlim(27, 39); a2.set_xlabel("Holdout RMSE (µg m$^{-3}$)"); a2.text(-0.02, 1.04, "b", transform=a2.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")
fig.tight_layout(); fig.savefig(FIG / "fig4_shap_vs_ablation.png"); plt.close(fig)

# Fig 5: error by observed AQI class
T = pd.read_csv(OUT / "r07_error_by_stratum.csv").iloc[:5]; x = np.arange(5); w = 0.38
names = ["Moderate\n9.1-35.4", "USG\n35.5-55.4", "Unhealthy\n55.5-125.4", "Very\nunhealthy\n125.5-225.4", "Hazardous\n> 225.4"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
a1.bar(x - w / 2, T.per_rmse, w, color=G, label="persistence"); a1.bar(x + w / 2, T.xgb_rmse, w, color=B, label="XGBoost"); a1.set_ylabel("RMSE (µg m$^{-3}$)"); a1.legend(frameon=False, fontsize=8)
a2.bar(x - w / 2, T.per_bias, w, color=G); a2.bar(x + w / 2, T.xgb_bias, w, color=B); a2.axhline(0, color=K, lw=0.7); a2.set_ylabel("Mean bias, forecast − observed (µg m$^{-3}$)")
for a in (a1, a2): a.set_xticks(x); a.set_xticklabels([f"{n}\n(n = {k})" for n, k in zip(names, T.n)], fontsize=8); a.set_xlabel("Observed class on day t+1 (µg m$^{-3}$)")
fig.tight_layout(); fig.savefig(FIG / "fig5_error_by_class.png"); plt.close(fig)
print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
