"""
run_baselines.py — Phase 4: evaluate the naive baselines on the SAME splits as the models.

These numbers are the bar every Phase-5 model must clear. We report each baseline's
RMSE/MAE/sMAPE/R² as mean ± std across the walk-forward folds, and (separately) on the
untouched 12-month hold-out. Skill scores in Phase 5 are computed relative to these.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from utils import load_config, set_global_seed, get_paths, ensure_dirs, get_logger
from validation import split_final_holdout, make_walk_forward
from evaluate import all_metrics
import baselines as bl

log = get_logger("baselines")


def _metrics_masked(y, pred):
    """Compute metrics on finite rows only (lag-7 can be NaN early in a series)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    m = np.isfinite(pred) & np.isfinite(y)
    out = all_metrics(y[m], pred[m]); out["n"] = int(m.sum())
    return out


def evaluate_walk_forward(dev: pd.DataFrame, cfg: dict) -> dict:
    """Per-fold baseline metrics over the expanding-window dev folds."""
    tscv = make_walk_forward(cfg, len(dev))
    y = dev["y_next_day"].to_numpy(float)
    acc = {"persistence": [], "climatology": [], "seasonal_naive_lag7": []}
    for tr_idx, te_idx in tscv.split(dev):
        tr, te = dev.iloc[tr_idx], dev.iloc[te_idx]
        yte = y[te_idx]
        acc["persistence"].append(_metrics_masked(yte, bl.persistence(te)))
        clim = bl.fit_climatology(tr)  # FIT ON TRAIN FOLD ONLY
        acc["climatology"].append(_metrics_masked(yte, bl.predict_climatology(clim, te)))
        acc["seasonal_naive_lag7"].append(_metrics_masked(yte, bl.seasonal_naive_lag7(te)))
    return acc


def summarise(acc: dict) -> pd.DataFrame:
    """Mean ± std of each metric across folds, per baseline."""
    rows = []
    for name, folds in acc.items():
        df = pd.DataFrame(folds)
        rows.append({"baseline": name,
                     **{f"{m}_mean": df[m].mean() for m in ["rmse", "mae", "smape", "r2"]},
                     **{f"{m}_std": df[m].std() for m in ["rmse", "mae", "smape", "r2"]}})
    return pd.DataFrame(rows)


def evaluate_holdout(dev: pd.DataFrame, holdout: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Baseline metrics on the untouched hold-out (climatology fit on ALL dev)."""
    y = holdout["y_next_day"].to_numpy(float)
    clim = bl.fit_climatology(dev)
    rows = [
        {"baseline": "persistence", **_metrics_masked(y, bl.persistence(holdout))},
        {"baseline": "climatology", **_metrics_masked(y, bl.predict_climatology(clim, holdout))},
        {"baseline": "seasonal_naive_lag7", **_metrics_masked(y, bl.seasonal_naive_lag7(holdout))},
    ]
    return pd.DataFrame(rows)


def main():
    """Run Phase 4: evaluate the naive baselines on the walk-forward folds and the
    untouched hold-out, write the report, and stash baseline RMSE for Phase-5 skill."""
    cfg = load_config()
    set_global_seed(cfg["project"]["random_seed"])
    ensure_dirs(cfg)
    paths = get_paths(cfg)

    table = pd.read_parquet(paths["processed_dir"] / "supervised_daily.parquet")
    dev, holdout = split_final_holdout(table, cfg)
    log.info("dev=%d holdout=%d (holdout %s..%s)", len(dev), len(holdout),
             holdout.index.min().date(), holdout.index.max().date())

    acc = evaluate_walk_forward(dev, cfg)
    cv = summarise(acc)
    ho = evaluate_holdout(dev, holdout, cfg)
    log.info("\nCV baselines:\n%s", cv.round(2).to_string(index=False))
    log.info("\nHold-out baselines:\n%s", ho.round(2).to_string(index=False))

    _write_report(paths, cfg, dev, holdout, acc, cv, ho)
    # stash CV+holdout baseline RMSE for Phase 5 skill scores
    persist_cv = float(np.mean([f["rmse"] for f in acc["persistence"]]))
    clim_cv = float(np.mean([f["rmse"] for f in acc["climatology"]]))
    (paths["processed_dir"] / "baseline_rmse.json").write_text(json.dumps({
        "cv": {"persistence": persist_cv, "climatology": clim_cv},
        "holdout": {r["baseline"]: r["rmse"] for _, r in ho.iterrows()}}, indent=1))
    log.info("Phase 4 complete.")


def _write_report(paths, cfg, dev, holdout, acc, cv, ho):
    """Write reports/phase4_baselines.md (CV mean±std, hold-out, and per-fold RMSE)."""
    L = ["# Phase 4 — Baselines (the bar to beat)\n",
         f"_Generated {date.today()}._ Splits: expanding-window walk-forward "
         f"({cfg['split']['walk_forward']['n_splits']} folds, gap "
         f"{cfg['split']['walk_forward']['gap_days']}) on dev (n={len(dev)}); untouched "
         f"12-month hold-out (n={len(holdout)}, {holdout.index.min().date()}.."
         f"{holdout.index.max().date()}).\n",
         "## Walk-forward CV (mean ± std across folds)\n",
         "| Baseline | RMSE | MAE | sMAPE % | R² |", "|---|---|---|---|---|"]
    for _, r in cv.iterrows():
        L.append(f"| {r['baseline']} | {r['rmse_mean']:.2f} ± {r['rmse_std']:.2f} | "
                 f"{r['mae_mean']:.2f} ± {r['mae_std']:.2f} | {r['smape_mean']:.1f} ± "
                 f"{r['smape_std']:.1f} | {r['r2_mean']:.3f} ± {r['r2_std']:.3f} |")
    L += ["\n## Untouched hold-out\n", "| Baseline | RMSE | MAE | sMAPE % | R² | n |",
          "|---|---|---|---|---|---|"]
    for _, r in ho.iterrows():
        L.append(f"| {r['baseline']} | {r['rmse']:.2f} | {r['mae']:.2f} | {r['smape']:.1f} | "
                 f"{r['r2']:.3f} | {int(r['n'])} |")
    L += ["\n## Per-fold RMSE\n", "| Fold | persistence | climatology | seasonal_naive_lag7 |",
          "|---|---|---|---|"]
    for i in range(len(acc["persistence"])):
        L.append(f"| {i+1} | {acc['persistence'][i]['rmse']:.2f} | "
                 f"{acc['climatology'][i]['rmse']:.2f} | "
                 f"{acc['seasonal_naive_lag7'][i]['rmse']:.2f} |")
    (paths["reports_dir"] / "phase4_baselines.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
