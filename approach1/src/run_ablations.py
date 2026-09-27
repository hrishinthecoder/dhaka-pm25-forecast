"""
run_ablations.py — Phase 8: which feature groups actually carry the forecast?

Each experiment in config.ablations.experiments is run through the IDENTICAL walk-forward
pipeline as Phase 5, with the LightGBM hyperparameters held FIXED at the Phase-5 best
values. Holding the model fixed and varying only the feature set isolates the marginal
contribution of each group — the cleanest ablation design.

The headline ablation is `omaq_pm2_5` (CAMS model PM2.5): we report core WITH vs WITHOUT it
and state the circularity caveat, because predicting measured PM2.5 from a model's PM2.5
estimate is near-circular and a reviewer will object if it is not disclosed.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from utils import load_config, set_global_seed, get_paths, ensure_dirs, get_logger
from validation import split_final_holdout
from evaluate import rmse, skill_score
from models_tree import cv_fold_metrics, fit_full, predict_bundle

log = get_logger("ablations")


def _columns_for_groups(group_names, feat_groups):
    """All feature columns whose group is in the requested set (skips empty groups)."""
    return [c for c, g in feat_groups.items() if g in set(group_names)]


def main():
    """Run Phase 8: evaluate every config-defined feature-group ablation through the same
    walk-forward + hold-out pipeline (LightGBM, fixed params), and write the comparison."""
    cfg = load_config()
    set_global_seed(cfg["project"]["random_seed"])
    ensure_dirs(cfg)
    paths = get_paths(cfg)

    table = pd.read_parquet(paths["processed_dir"] / "supervised_daily.parquet")
    feat_info = json.loads((paths["processed_dir"] / "feature_groups.json").read_text())
    feat_groups = feat_info["groups"]
    params = json.loads((paths["artifacts_dir"] / "lgbm_best_params.json").read_text())

    dev, holdout = split_final_holdout(table, cfg)
    y_dev = dev["y_next_day"].to_numpy(float)
    y_ho = holdout["y_next_day"].to_numpy(float)
    base = json.loads((paths["processed_dir"] / "baseline_rmse.json").read_text())
    pc, hp = base["cv"]["persistence"], base["holdout"]["persistence"]

    rows = []
    for exp in cfg["ablations"]["experiments"]:
        cols = _columns_for_groups(exp["groups"], feat_groups)
        if not cols:
            log.warning("experiment %s has no columns, skipping", exp["name"])
            continue
        cv = cv_fold_metrics("lightgbm", params, dev[cols], y_dev, cfg)
        bundle = fit_full("lightgbm", params, dev[cols], y_dev, cfg)
        ho_pred = predict_bundle(bundle, holdout[cols])
        rows.append({
            "experiment": exp["name"], "n_features": len(cols),
            "cv_rmse": cv["rmse"].mean(), "cv_rmse_std": cv["rmse"].std(),
            "cv_mae": cv["mae"].mean(), "cv_r2": cv["r2"].mean(),
            "ho_rmse": rmse(y_ho, ho_pred), "ho_r2": float(np.corrcoef(y_ho, ho_pred)[0, 1] ** 2),
            "skill_persist_cv": skill_score(cv["rmse"].mean(), pc),
            "skill_persist_ho": skill_score(rmse(y_ho, ho_pred), hp),
        })
        log.info("%-32s nfeat=%2d  CV RMSE=%.2f  HO RMSE=%.2f", exp["name"],
                 len(cols), rows[-1]["cv_rmse"], rows[-1]["ho_rmse"])

    res = pd.DataFrame(rows)
    _write_report(paths, res, pc, hp)
    _update_memory(paths, res)
    log.info("Phase 8 complete.")


def _write_report(paths, res, pc, hp):
    """Write reports/phase8_ablations.md: the ablation table plus the omaq_pm2_5 caveat."""
    L = ["# Phase 8 — Ablations (which signals carry the forecast)\n",
         f"_Generated {date.today()}._ LightGBM with FIXED Phase-5 best params; only the "
         "feature set varies. Same expanding walk-forward folds + untouched hold-out. "
         f"Baseline bar: persistence CV RMSE {pc:.2f}, hold-out {hp:.2f}.\n",
         "| Experiment | #feat | CV RMSE | CV R² | HO RMSE | skill vs persist (CV) |",
         "|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {r['experiment']} | {int(r['n_features'])} | "
                 f"{r['cv_rmse']:.2f} ± {r['cv_rmse_std']:.2f} | {r['cv_r2']:.3f} | "
                 f"{r['ho_rmse']:.2f} | {r['skill_persist_cv']:+.3f} |")

    # omaq_pm2_5 with-vs-without comparison (core vs core_plus_cams_pm25)
    core = res[res["experiment"] == "core"]
    cams = res[res["experiment"] == "core_plus_cams_pm25_CAVEATED"]
    L.append("\n## The `omaq_pm2_5` ablation (circularity caveat)\n")
    if len(core) and len(cams):
        d = core.iloc[0]["cv_rmse"] - cams.iloc[0]["cv_rmse"]
        L.append(f"- core CV RMSE **{core.iloc[0]['cv_rmse']:.2f}** vs "
                 f"core+omaq_pm2_5 CV RMSE **{cams.iloc[0]['cv_rmse']:.2f}** "
                 f"(Δ {d:+.2f}; positive Δ = the CAMS PM2.5 feature lowers error).\n")
    L.append("> **Circularity caveat:** `omaq_pm2_5` is a CAMS/Open-Meteo MODEL OUTPUT of "
             "PM2.5 (hourly r≈0.754 with measured PM2.5, ~86% missing). Predicting *measured* "
             "PM2.5 from a model's PM2.5 estimate is near-circular, so any gain it brings is "
             "**not** evidence of a better forecaster. It is reported here for disclosure only "
             "and is EXCLUDED from the default/core model.\n")
    L.append("\n## Interpretation\n- Marginal group contributions are read by walking "
             "met_only → +calendar → +lags/rolling → +fire (core). The largest CV-RMSE drop "
             "marks the signal that carries the forecast.")
    (paths["reports_dir"] / "phase8_ablations.md").write_text("\n".join(L), encoding="utf-8")


def _update_memory(paths, res):
    """Append the best-feature-set ablation result to the decisions log."""
    best = res.loc[res["cv_rmse"].idxmin()]
    with open(paths["memory_dir"] / "architecture_decisions.md", "a", encoding="utf-8") as fh:
        fh.write(f"\n## {date.today()} — Phase 8 (ablations)\n"
                 f"- Best CV-RMSE feature set: {best['experiment']} ({best['cv_rmse']:.2f}). "
                 f"omaq_pm2_5 ablation reported with circularity caveat; excluded from core.\n")


if __name__ == "__main__":
    main()
