"""
run_shap.py — Phase 7: explain the LightGBM forecast with SHAP.

QUESTION WE WANT ANSWERED
    Where does the skill come from — PM2.5 autoregressive lags (persistence-like memory),
    meteorology (boundary-layer height, wind, humidity), or fire activity? SHAP attributes
    each prediction to its features; summing mean|SHAP| within each feature GROUP tells us
    which physical signal carries the forecast — the interpretation reviewers ask for.

SHAP = SHapley Additive exPlanations: each feature's contribution to pushing a prediction
above/below the average, with a fair (game-theoretic) split of credit. TreeExplainer
computes it exactly and fast for gradient-boosted trees.
"""

from __future__ import annotations

import json
from datetime import date

import joblib
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from utils import load_config, set_global_seed, get_paths, ensure_dirs, get_logger
from validation import split_final_holdout
from models_tree import _apply_medians

log = get_logger("shap")


def main():
    """Run Phase 7: compute SHAP attributions for the LightGBM model, aggregate them by
    feature group, save beeswarm/bar/group figures, and write the interpretation report."""
    cfg = load_config()
    set_global_seed(cfg["project"]["random_seed"])
    ensure_dirs(cfg)
    paths = get_paths(cfg)

    bundle = joblib.load(paths["artifacts_dir"] / "model_lightgbm.joblib")
    feat_info = json.loads((paths["processed_dir"] / "feature_groups.json").read_text())
    groups = feat_info["groups"]

    table = pd.read_parquet(paths["processed_dir"] / "supervised_daily.parquet")
    dev, _ = split_final_holdout(table, cfg)

    # Rebuild the exact matrix the model was trained on: median-impute + anomaly flag.
    X = dev[bundle["feature_cols"]]
    X_imp = _apply_medians(X, bundle["medians"])
    if bundle["iso"] is not None:
        X_imp = X_imp.assign(anomaly_flag=(bundle["iso"].predict(X_imp) == -1).astype(int))
    else:
        X_imp = X_imp.assign(anomaly_flag=0)

    explainer = shap.TreeExplainer(bundle["model"])
    sv = explainer.shap_values(X_imp)
    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.Series(mean_abs, index=X_imp.columns).sort_values(ascending=False)

    # ---- group attribution: sum mean|SHAP| by feature group ----
    grp = {}
    for feat, val in imp.items():
        g = groups.get(feat, "outlier_flag" if feat == "anomaly_flag" else "other")
        grp[g] = grp.get(g, 0.0) + float(val)
    grp_series = pd.Series(grp).sort_values(ascending=False)
    grp_pct = (grp_series / grp_series.sum() * 100).round(1)

    # ---- figures ----
    shap.summary_plot(sv, X_imp, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(paths["figures_dir"] / "shap_beeswarm.png", dpi=130); plt.close()
    shap.summary_plot(sv, X_imp, plot_type="bar", show=False, max_display=20)
    plt.tight_layout(); plt.savefig(paths["figures_dir"] / "shap_bar.png", dpi=130); plt.close()
    plt.figure(figsize=(7, 4))
    grp_series.iloc[::-1].plot.barh()
    plt.xlabel("Σ mean|SHAP|"); plt.title("Feature-group attribution (LightGBM)")
    plt.tight_layout(); plt.savefig(paths["figures_dir"] / "shap_group_attribution.png", dpi=130); plt.close()

    _write_report(paths, imp, grp_series, grp_pct)
    with open(paths["memory_dir"] / "architecture_decisions.md", "a", encoding="utf-8") as fh:
        top_grp = grp_pct.index[0]
        fh.write(f"\n## {date.today()} — Phase 7 (SHAP)\n- Group attribution (% of "
                 f"Σmean|SHAP|): {grp_pct.to_dict()}. Dominant signal: {top_grp}.\n")
    log.info("Phase 7 complete. Group attribution:\n%s", grp_pct.to_string())


def _write_report(paths, imp, grp_series, grp_pct):
    """Write reports/phase7_results.md: group attribution, top features, interpretation."""
    L = ["# Phase 7 — Explainability (SHAP on LightGBM)\n",
         f"_Generated {date.today()}._  TreeExplainer on the dev set.\n",
         "## Feature-group attribution (which signal carries the forecast)\n",
         "| Group | Σ mean\\|SHAP\\| | share % |", "|---|---|---|"]
    for g in grp_series.index:
        L.append(f"| {g} | {grp_series[g]:.3f} | {grp_pct[g]:.1f}% |")
    L += ["\n_Figure: `figures/shap_group_attribution.png`._\n",
          "\n## Top 20 individual features (mean\\|SHAP\\|)\n",
          "| Rank | Feature | mean\\|SHAP\\| |", "|---|---|---|"]
    for i, (f, v) in enumerate(imp.head(20).items(), 1):
        L.append(f"| {i} | `{f}` | {v:.3f} |")
    L += ["\n_Figures: `figures/shap_beeswarm.png`, `figures/shap_bar.png`._\n",
          "\n## Interpretation\n",
          f"The forecast is dominated by **{grp_pct.index[0]}** "
          f"({grp_pct.iloc[0]:.0f}% of total attribution), followed by "
          f"**{grp_pct.index[1]}** ({grp_pct.iloc[1]:.0f}%). A lag-dominated model is "
          "persistence-like memory; meteorology dominance (boundary-layer height, wind, "
          "humidity) is the physically interpretable signal reviewers want; fire "
          "contribution flags biomass-burning episodes. See the group table above for the "
          "actual split on this run."]
    (paths["reports_dir"] / "phase7_results.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
