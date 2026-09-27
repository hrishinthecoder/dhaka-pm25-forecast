"""Rolling-origin multi-year backtest (supplementary analysis).

Defeats the "your test is one thin window (n=219)" critique by re-running the EXACT
main-pipeline protocol (same feature builder, tuning grids, seed, metrics) across
six expanding-window folds:

    for y in [2019, 2020, 2021, 2022, 2023, 2024]:
        train = valid (t→t+1) pairs with day-t ≤ Dec 31 of year y
        test  = valid pairs whose day-t falls in year y+1
        tune  = TimeSeriesSplit(5) over the fold's train window; refit on train

The final fold (y=2024 → test 2025) is a thin, high-variance fold (n≈75) because the
single-station target (sensor 24434) ends 2025-03-24.
Nothing here re-tunes grids or changes seeds — it imports and reuses
`src.features.daily`, `src.models.train`, `src.evaluate.metrics`.

    python -m src.evaluate.rolling_backtest --config config/model.yaml

Outputs reports/rolling_backtest.csv and reports/figures/rolling_backtest_rmse.png.
Runtime: ~6 folds × (RF + 2 grid searches × TimeSeriesSplit(5)); a few minutes on
a laptop. Per-fold progress is logged to stdout.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.features.daily import build_daily, model_matrix
from src.models.train import apply_scaler, baselines, fit_scaler, train_models
from src.evaluate.metrics import (diebold_mariano, paired_bootstrap_diff,
                                  reg_metrics)

FOLDS = [2019, 2020, 2021, 2022, 2023, 2024]   # train ≤ Dec 31 y ; test y+1
ML_MODELS = ["random_forest", "xgboost", "lightgbm"]
ALL_MODELS = ["persistence", "climatology"] + ML_MODELS


def _fold(feat, features, paired, cfg, y, seed) -> list[dict]:
    end = pd.Timestamp(f"{y}-12-31")
    X, target = model_matrix(feat, features)
    tr = X.index <= end
    te = X.index.year == (y + 1)
    X_tr, y_tr = X[tr], target[tr]
    X_te, y_te = X[te], target[te]
    if len(X_te) == 0 or len(X_tr) == 0:
        return []

    p_tr = paired[paired.index <= end]
    p_te = paired[paired.index.year == (y + 1)].reindex(X_te.index)

    scaler = fit_scaler(X_tr)
    Xtr_s, Xte_s = apply_scaler(scaler, X_tr), apply_scaler(scaler, X_te)

    preds = baselines(p_tr, p_te)                       # persistence, climatology
    # Tune over the fold's train window (TimeSeriesSplit 5) and refit on train.
    models, _ = train_models(Xtr_s, y_tr.values, Xtr_s, y_tr.values, cfg)
    for name, mdl in models.items():
        preds[name] = mdl.predict(Xte_s)

    yv = y_te.values
    persist = preds["persistence"]
    rows = []
    for name in ALL_MODELS:
        if name not in preds:
            continue
        m = reg_metrics(yv, preds[name])
        row = {"fold_train_end": y, "test_year": y + 1, "model": name,
               "n_test": m["n"], "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"],
               "delta_rmse": np.nan, "delta_rmse_ci_lo": np.nan,
               "delta_rmse_ci_hi": np.nan, "delta_mae": np.nan, "dm_pvalue": np.nan}
        if name != "persistence":
            b = paired_bootstrap_diff(yv, preds[name], persist, n_boot=1000, seed=seed)
            dm = diebold_mariano(yv, preds[name], persist, h=1)
            row.update({"delta_rmse": b["delta_rmse"],
                        "delta_rmse_ci_lo": b["delta_rmse_ci_lo"],
                        "delta_rmse_ci_hi": b["delta_rmse_ci_hi"],
                        "delta_mae": b["delta_mae"], "dm_pvalue": dm["dm_pvalue"]})
        rows.append(row)
    return rows


def _figure(df: pd.DataFrame, out: Path) -> None:
    xgb = df[df["model"] == "xgboost"].sort_values("test_year")
    per = df[df["model"] == "persistence"].sort_values("test_year")
    yr = xgb["test_year"].values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(yr, xgb["rmse"], "o-", color="#1f77b4", label="XGBoost")
    ax1.plot(per["test_year"], per["rmse"], "s--", color="#888", label="persistence")
    ax1.set_ylabel("test RMSE (µg/m³)")
    ax1.set_title("Rolling-origin backtest — XGBoost vs persistence")
    ax1.legend(); ax1.grid(alpha=0.3)

    d = xgb["delta_rmse"].values
    lo = d - xgb["delta_rmse_ci_lo"].values
    hi = xgb["delta_rmse_ci_hi"].values - d
    ax2.errorbar(yr, d, yerr=[lo, hi], fmt="o", color="#d62728",
                 capsize=4, label="ΔRMSE = XGB − persistence (95% CI)")
    ax2.axhline(0, color="k", lw=1)
    ax2.set_ylabel("ΔRMSE (µg/m³)"); ax2.set_xlabel("test year")
    ax2.set_title("Negative ⇒ XGBoost better; CI excluding 0 ⇒ beats persistence")
    ax2.grid(alpha=0.3); ax2.legend()
    ax2.set_xticks(yr)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def run(config_path: str = "config/model.yaml") -> pd.DataFrame:
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = cfg["artifacts"]["random_seed"]
    np.random.seed(seed)
    reports = Path(cfg["artifacts"]["reports_dir"])
    figs = reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    data_path = cfg.get("data", {}).get("input_path",
                                        "data/processed/dhaka_aq_master_singlestation.parquet")
    df = pd.read_parquet(data_path)
    feat, features = build_daily(df, cfg)
    paired = feat[feat["pair_valid"]].dropna(subset=features + ["y_next"])

    rows: list[dict] = []
    for y in FOLDS:
        t0 = time.time()
        fr = _fold(feat, features, paired, cfg, y, seed)
        rows += fr
        nt = fr[0]["n_test"] if fr else 0
        print(f"[backtest] fold train<={y} / test {y+1}: n_test={nt} "
              f"({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_csv(reports / "rolling_backtest.csv", index=False)
    _figure(out, figs / "rolling_backtest_rmse.png")
    print("[backtest] wrote reports/rolling_backtest.csv + figures/rolling_backtest_rmse.png")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/model.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
