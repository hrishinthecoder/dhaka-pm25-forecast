"""
run_models.py — Phase 5: gradient boosting (the headline result).

WHAT IT DOES
    For LightGBM and XGBoost: tune hyperparameters with Optuna using the expanding-window
    walk-forward CV RMSE (the hold-out is NEVER passed to Optuna), then report per-fold CV
    metrics, fit a final model on ALL dev data, and score the untouched 12-month hold-out
    ONCE. RandomForest is included as a lightly-configured secondary tree baseline.

    Every model is reported as RMSE/MAE/sMAPE/R² (mean ± std across folds) AND as a skill
    score over persistence and climatology. A model only "wins" if it beats persistence.

LEAKAGE CONTROLS (all enforced in models_tree.py)
    - Splitter is sklearn TimeSeriesSplit = expanding-window walk-forward; never KFold,
      never shuffled; a gap separates train from test.
    - Imputer medians + IsolationForest flag are fit on the fold's TRAIN rows only.
    - Early stopping uses an inner time-ordered slice of the train fold (not the test fold).
    - The hold-out is split off first and only scored after all tuning is done.
"""

from __future__ import annotations

import json
from datetime import date

import joblib
import pandas as pd

from utils import load_config, set_global_seed, get_paths, ensure_dirs, get_logger
from validation import split_final_holdout
from evaluate import all_metrics, skill_score
import models_tree as mt

log = get_logger("models")


def main():
    """Run Phase 5: tune LightGBM/XGBoost with Optuna on the walk-forward CV, fit each
    model on dev, score the untouched hold-out once, save artifacts, and write the report."""
    cfg = load_config()
    seed = set_global_seed(cfg["project"]["random_seed"])
    ensure_dirs(cfg)
    paths = get_paths(cfg)
    log.info("seed=%s", seed)

    table = pd.read_parquet(paths["processed_dir"] / "supervised_daily.parquet")
    feat_info = json.loads((paths["processed_dir"] / "feature_groups.json").read_text())
    core_cols = feat_info["core_cols"]

    dev, holdout = split_final_holdout(table, cfg)
    X_dev, y_dev = dev[core_cols], dev["y_next_day"].to_numpy(float)
    X_ho, y_ho = holdout[core_cols], holdout["y_next_day"].to_numpy(float)
    base = json.loads((paths["processed_dir"] / "baseline_rmse.json").read_text())
    log.info("dev=%d holdout=%d core_features=%d", len(dev), len(holdout), len(core_cols))

    results = {}   # name -> {cv_df, best_params, holdout_metrics}
    for name in ["lightgbm", "xgboost", "random_forest"]:
        if not cfg["models"][name]["enabled"]:
            continue
        log.info("=== %s ===", name)
        if name in ("lightgbm", "xgboost"):
            best_params, study = mt.tune(name, X_dev, y_dev, cfg)
            log.info("%s best CV RMSE=%.3f params=%s", name, study.best_value, best_params)
        else:
            best_params = {}  # RF uses fixed config params
        cv_df = mt.cv_fold_metrics(name, best_params, X_dev, y_dev, cfg)
        bundle = mt.fit_full(name, best_params, X_dev, y_dev, cfg)
        ho_pred = mt.predict_bundle(bundle, X_ho)
        ho_metrics = all_metrics(y_ho, ho_pred)
        joblib.dump(bundle, paths["artifacts_dir"] / f"model_{name}.joblib")
        results[name] = {"cv": cv_df, "best_params": best_params, "holdout": ho_metrics}
        log.info("%s CV RMSE=%.2f±%.2f | holdout RMSE=%.2f R²=%.3f", name,
                 cv_df["rmse"].mean(), cv_df["rmse"].std(),
                 ho_metrics["rmse"], ho_metrics["r2"])

    _write_report(paths, cfg, dev, holdout, results, base)
    _update_memory(paths, results, base)
    log.info("Phase 5 complete.")


def _skill_line(rmse_val, base_rmse):
    """Skill score of a model RMSE vs a baseline RMSE (positive = beats the baseline)."""
    return skill_score(rmse_val, base_rmse)


def _write_report(paths, cfg, dev, holdout, results, base):
    """Write reports/phase5_models.md: CV/hold-out tables, skill scores, params, verdict."""
    pc, cc = base["cv"]["persistence"], base["cv"]["climatology"]
    hp = base["holdout"]["persistence"]
    L = ["# Phase 5 — Gradient Boosting (headline models)\n",
         f"_Generated {date.today()}._\n",
         "**Validation:** scikit-learn `TimeSeriesSplit` = expanding-window walk-forward "
         f"({cfg['split']['walk_forward']['n_splits']} folds, gap "
         f"{cfg['split']['walk_forward']['gap_days']}). No KFold, no shuffling — test fold "
         "always strictly after train. Optuna optimised the **walk-forward CV RMSE only**; "
         f"the 12-month hold-out (n={len(holdout)}, {holdout.index.min().date()}.."
         f"{holdout.index.max().date()}) was never seen during tuning. Dev n={len(dev)}.\n",
         f"\nBaseline bar — CV: persistence RMSE **{pc:.2f}**, climatology **{cc:.2f}**; "
         f"hold-out persistence RMSE **{hp:.2f}**.\n",
         "## Walk-forward CV (mean ± std across folds) + skill vs persistence\n",
         "| Model | RMSE | MAE | sMAPE % | R² | skill vs persist | skill vs clim |",
         "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        cv = r["cv"]
        rm = cv["rmse"].mean()
        L.append(f"| {name} | {rm:.2f} ± {cv['rmse'].std():.2f} | "
                 f"{cv['mae'].mean():.2f} ± {cv['mae'].std():.2f} | "
                 f"{cv['smape'].mean():.1f} ± {cv['smape'].std():.1f} | "
                 f"{cv['r2'].mean():.3f} ± {cv['r2'].std():.3f} | "
                 f"{_skill_line(rm, pc):+.3f} | {_skill_line(rm, cc):+.3f} |")
    L += ["\n## Untouched hold-out (scored once) + skill vs persistence\n",
          "| Model | RMSE | MAE | sMAPE % | R² | skill vs persist |",
          "|---|---|---|---|---|---|"]
    for name, r in results.items():
        m = r["holdout"]
        L.append(f"| {name} | {m['rmse']:.2f} | {m['mae']:.2f} | {m['smape']:.1f} | "
                 f"{m['r2']:.3f} | {_skill_line(m['rmse'], hp):+.3f} |")
    L += ["\n## Per-fold CV RMSE\n", "| Fold | " +
          " | ".join(results.keys()) + " |", "|---|" + "---|" * len(results)]
    n_folds = len(next(iter(results.values()))["cv"])
    for i in range(n_folds):
        L.append(f"| {i+1} | " +
                 " | ".join(f"{results[n]['cv']['rmse'].iloc[i]:.2f}" for n in results) + " |")
    L += ["\n## Best Optuna params\n"]
    for name, r in results.items():
        if r["best_params"]:
            L.append(f"- **{name}**: `{r['best_params']}`")
    L += ["\n## Verdict\n"]
    winners = [n for n, r in results.items() if _skill_line(r['cv']['rmse'].mean(), pc) > 0]
    if winners:
        L.append(f"Models beating persistence on CV RMSE: **{', '.join(winners)}**.")
    else:
        L.append("⚠️ No model beat persistence on CV RMSE — STOP and rethink features/target "
                 "per the master prompt.")
    (paths["reports_dir"] / "phase5_models.md").write_text("\n".join(L), encoding="utf-8")


def _update_memory(paths, results, base):
    """Append Phase 4/5 results and the best model to the decisions log + state tracker."""
    mem = paths["memory_dir"]
    pc = base["cv"]["persistence"]; hp = base["holdout"]["persistence"]
    best = min(results.items(), key=lambda kv: kv[1]["cv"]["rmse"].mean())
    entry = [f"\n## {date.today()} — Phases 4 & 5\n",
             f"- Baselines (CV RMSE): persistence {pc:.2f}, climatology "
             f"{base['cv']['climatology']:.2f}.\n"]
    for name, r in results.items():
        entry.append(f"- {name}: CV RMSE {r['cv']['rmse'].mean():.2f}, holdout RMSE "
                     f"{r['holdout']['rmse']:.2f} (R² {r['holdout']['r2']:.3f}), "
                     f"skill vs persist (CV) {skill_score(r['cv']['rmse'].mean(), pc):+.3f}.\n")
    entry.append(f"- Best model: **{best[0]}**. Validation = expanding walk-forward "
                 "TimeSeriesSplit; hold-out untouched by Optuna.\n")
    with open(mem / "architecture_decisions.md", "a", encoding="utf-8") as fh:
        fh.write("".join(entry))
    state = {"phase": 5, "status": "complete",
             "last_artifact": "artifacts/model_*.joblib",
             "best_model": best[0],
             "best_cv_rmse": round(float(best[1]["cv"]["rmse"].mean()), 3),
             "persistence_cv_rmse": round(pc, 3),
             "open_issues": ["N≈1746 small-sample (accepted)"]}
    (mem / "state_tracker.json").write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
