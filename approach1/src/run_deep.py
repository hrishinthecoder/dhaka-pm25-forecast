"""
run_deep.py — Phase 6: Bi-LSTM+attention vs LightGBM (honest, same protocol).

Builds 30-day look-back sequences of per-day features, runs them through the SAME
expanding-window walk-forward folds and the SAME untouched hold-out as the trees, averages
over 3 seeds (small models are seed-sensitive), and writes a head-to-head report with the
sample-size caveat. No Optuna here — fixed config budget, by design (falsification test).
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from utils import load_config, set_global_seed, get_paths, ensure_dirs, get_logger
from data_loader import load_master, load_target_sensor
from targets import daily_mean_target
from features import _daily_aggregate_master
from validation import split_final_holdout, make_walk_forward
from evaluate import all_metrics, skill_score
from models_deep import train_one

log = get_logger("deep")
LOOKBACK = None  # set from config in main


def build_daily_base(cfg) -> pd.DataFrame:
    """Per-day feature frame for the sequence model (continuous daily calendar).

    Columns: pm25_daily_mean (single-station), ERA5 daily means, FIRMS, and calendar
    (sin/cos of month/weekday/day-of-year + weekend/holiday) for THAT day. All values are
    knowable on their own day; the look-back window for origin t ends at day t, so nothing
    after t enters a sample.
    """
    master = load_master(cfg)
    daily = daily_mean_target(load_target_sensor(cfg), cfg)
    dm = _daily_aggregate_master(master, cfg)

    idx = pd.date_range(min(dm.index.min(), daily.index.min()),
                        max(dm.index.max(), daily.index.max()), freq="D")
    base = pd.DataFrame(index=idx)
    base["pm25_daily_mean"] = daily["pm25_daily_mean"].reindex(idx)
    for c in cfg["features"]["era5_meteorology"]:
        base[c] = dm[c].reindex(idx)
    for c in ["firms_count", "firms_frp_sum"]:
        if c in dm.columns:
            base[c] = dm[c].reindex(idx)
    # calendar of each day (deterministic)
    base["month_sin"] = np.sin(2*np.pi*idx.month/12); base["month_cos"] = np.cos(2*np.pi*idx.month/12)
    base["dow_sin"] = np.sin(2*np.pi*idx.dayofweek/7); base["dow_cos"] = np.cos(2*np.pi*idx.dayofweek/7)
    base["doy_sin"] = np.sin(2*np.pi*idx.dayofyear/366); base["doy_cos"] = np.cos(2*np.pi*idx.dayofyear/366)
    base["is_weekend"] = idx.dayofweek.isin([4, 5]).astype(int)
    return base


def make_sequences(origin_days, base: pd.DataFrame, lookback: int):
    """Stack a (lookback, F) window ending at each origin day t -> (n, lookback, F)."""
    F = base.shape[1]
    arr = base.values.astype("float64")
    pos = {d: i for i, d in enumerate(base.index)}
    out = np.full((len(origin_days), lookback, F), np.nan)
    for j, t in enumerate(origin_days):
        end = pos[t]
        start = end - lookback + 1
        if start < 0:
            window = arr[0:end + 1]
            out[j, lookback - window.shape[0]:] = window  # left-pad with NaN
        else:
            out[j] = arr[start:end + 1]
    return out


def main():
    """Run Phase 6: build 30-day sequences, train the Bi-LSTM+attention on the same
    walk-forward folds and hold-out (seed-averaged), and write the honest comparison."""
    cfg = load_config()
    set_global_seed(cfg["project"]["random_seed"])
    ensure_dirs(cfg)
    paths = get_paths(cfg)
    lookback = cfg["deep_model"]["lookback_days"]
    seeds = list(range(cfg["deep_model"]["n_seeds_for_reporting"]))

    table = pd.read_parquet(paths["processed_dir"] / "supervised_daily.parquet")
    dev, holdout = split_final_holdout(table, cfg)
    base = build_daily_base(cfg)

    Xdev = make_sequences(list(dev.index), base, lookback)
    Xho = make_sequences(list(holdout.index), base, lookback)
    ydev = dev["y_next_day"].to_numpy(float)
    yho = holdout["y_next_day"].to_numpy(float)
    log.info("deep sequences: dev %s holdout %s (lookback=%d, F=%d)",
             Xdev.shape, Xho.shape, lookback, Xdev.shape[2])

    # ---- walk-forward CV (avg over seeds per fold) ----
    tscv = make_walk_forward(cfg, len(dev))
    fold_metrics = []
    for fold, (tr, te) in enumerate(tscv.split(Xdev), 1):
        seed_preds = [train_one(Xdev[tr], ydev[tr], Xdev[te], cfg, s) for s in seeds]
        pred = np.mean(seed_preds, axis=0)
        m = all_metrics(ydev[te], pred); m["fold"] = fold
        fold_metrics.append(m)
        log.info("fold %d deep RMSE=%.2f", fold, m["rmse"])
    cv = pd.DataFrame(fold_metrics)

    # ---- hold-out (avg over seeds, trained on full dev) ----
    ho_preds = [train_one(Xdev, ydev, Xho, cfg, s) for s in seeds]
    ho_pred = np.mean(ho_preds, axis=0)
    ho_m = all_metrics(yho, ho_pred)
    log.info("deep holdout RMSE=%.2f R2=%.3f", ho_m["rmse"], ho_m["r2"])

    _write_report(paths, cfg, dev, holdout, cv, ho_m)
    _update_memory(paths, cv, ho_m)
    log.info("Phase 6 complete.")


def _load_lgbm_numbers(paths):
    """Pull LightGBM CV/holdout RMSE from the Phase-5 state for a direct comparison."""
    try:
        import re
        txt = (paths["reports_dir"] / "phase5_models.md").read_text(encoding="utf-8")
        return txt
    except Exception:
        return ""


def _write_report(paths, cfg, dev, holdout, cv, ho_m):
    """Write reports/phase6_deep_comparison.md with the sample-size caveat + head-to-head."""
    base = json.loads((paths["processed_dir"] / "baseline_rmse.json").read_text())
    pc, hp = base["cv"]["persistence"], base["holdout"]["persistence"]
    L = ["# Phase 6 — Deep Comparison (Bi-LSTM + attention): falsification test\n",
         f"_Generated {date.today()}._\n",
         "> **Caveat (write this into the manuscript):** N≈1,746 supervised daily samples is "
         "tiny for a sequence model. This Bi-LSTM+attention is an honest comparison arm, not "
         "the headline. Same target, same expanding-window walk-forward folds, same untouched "
         f"hold-out, fixed config budget ({cfg['deep_model']['n_seeds_for_reporting']} seeds, "
         "no Optuna over-search). It is expected to be outperformed by LightGBM.\n",
         f"\nLook-back {cfg['deep_model']['lookback_days']} days; hidden "
         f"{cfg['deep_model']['hidden_size']}; heads {cfg['deep_model']['attention_heads']}; "
         f"dropout {cfg['deep_model']['dropout']}; early stopping patience "
         f"{cfg['deep_model']['early_stopping_patience']}.\n",
         "## Walk-forward CV (mean ± std over folds; seed-averaged predictions)\n",
         "| Metric | Bi-LSTM+attn |", "|---|---|",
         f"| RMSE | {cv['rmse'].mean():.2f} ± {cv['rmse'].std():.2f} |",
         f"| MAE | {cv['mae'].mean():.2f} ± {cv['mae'].std():.2f} |",
         f"| sMAPE % | {cv['smape'].mean():.1f} ± {cv['smape'].std():.1f} |",
         f"| R² | {cv['r2'].mean():.3f} ± {cv['r2'].std():.3f} |",
         f"| skill vs persistence | {skill_score(cv['rmse'].mean(), pc):+.3f} |",
         "\n## Untouched hold-out (scored once)\n",
         f"- RMSE **{ho_m['rmse']:.2f}**, MAE {ho_m['mae']:.2f}, sMAPE {ho_m['smape']:.1f}%, "
         f"R² {ho_m['r2']:.3f}, skill vs persistence {skill_score(ho_m['rmse'], hp):+.3f}.\n",
         "## Per-fold CV RMSE\n", "| Fold | RMSE |", "|---|---|"]
    for _, r in cv.iterrows():
        L.append(f"| {int(r['fold'])} | {r['rmse']:.2f} |")
    L += ["\n## Head-to-head (CV RMSE, lower = better)\n",
          "| Model | CV RMSE | Holdout RMSE |", "|---|---|---|",
          f"| **LightGBM (primary)** | 30.26 | 30.25 |",
          f"| Bi-LSTM+attention | {cv['rmse'].mean():.2f} | {ho_m['rmse']:.2f} |",
          f"| persistence baseline | {pc:.2f} | {hp:.2f} |",
          "\n**Interpretation:** see the verdict line vs LightGBM (30.26 CV / 30.25 holdout). "
          "If the deep model is higher (worse), that is the expected null result at this N and "
          "is reported as a finding, not hidden."]
    (paths["reports_dir"] / "phase6_deep_comparison.md").write_text("\n".join(L), encoding="utf-8")


def _update_memory(paths, cv, ho_m):
    """Append the deep-vs-tree comparison summary to the decisions log."""
    entry = (f"\n## {date.today()} — Phase 6 (deep falsification test)\n"
             f"- Bi-LSTM+attention: CV RMSE {cv['rmse'].mean():.2f}±{cv['rmse'].std():.2f}, "
             f"holdout RMSE {ho_m['rmse']:.2f} (R² {ho_m['r2']:.3f}). "
             f"Compared honestly to LightGBM (30.26 CV / 30.25 holdout). Fixed budget, 3 seeds.\n")
    with open(paths["memory_dir"] / "architecture_decisions.md", "a", encoding="utf-8") as fh:
        fh.write(entry)


if __name__ == "__main__":
    main()
