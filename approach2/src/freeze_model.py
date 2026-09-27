"""Freeze the published model: refit XGBoost (selected model, chosen
hyperparameters) on TRAIN ONLY (≤2023) and write the immutable artifact pack.

    python -m src.freeze_model --config config/model.yaml

Reuses the exact main-pipeline code path (same feature builder, tuning grids,
seed) so the frozen model is identical to the headline XGBoost. Writes:

    models/final/xgboost_final.json     native booster
    models/final/xgboost_final.joblib   sklearn wrapper
    models/final/scaler.joblib          StandardScaler (fit on train only)
    models/final/feature_list.json      ordered 45 feature names
    models/final/hyperparameters.json   tuned + fixed hyperparameters
    models/final/SHA256SUMS             checksums of all of the above + the parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.features.daily import build_daily, holdout_start, model_matrix, split_masks
from src.models.train import apply_scaler, fit_scaler, train_models

FINAL = Path("models/final")
PARQUET = "data/processed/dhaka_aq_master_singlestation.parquet"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(config_path: str = "config/model.yaml") -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = cfg["artifacts"]["random_seed"]
    np.random.seed(seed)
    FINAL.mkdir(parents=True, exist_ok=True)

    parquet = cfg.get("data", {}).get("input_path", PARQUET)
    df = pd.read_parquet(parquet)
    feat, features = build_daily(df, cfg)
    X, y = model_matrix(feat, features)
    tr, va, _ = split_masks(X.index, cfg)          # train = full pre-holdout window
    X_tr, y_tr = X[tr], y[tr]
    X_cv = pd.concat([X_tr, X[va]]).sort_index()
    y_cv = pd.concat([y_tr, y[va]]).sort_index()
    hs = holdout_start(X.index, cfg)
    trained_desc = (f"valid (t→t+1) pairs with day-t < {hs.date()} "
                    f"(pre-holdout; holdout = last {cfg['split'].get('holdout_months', 12)} "
                    "months of available pairs)") if hs is not None else \
                   f"valid (t→t+1) pairs with day-t <= {cfg['split']['train_end']}"

    scaler = fit_scaler(X_tr)                                  # TRAIN ONLY
    Xtr_s = apply_scaler(scaler, X_tr)
    Xcv_s = apply_scaler(scaler, X_cv)
    models, tuning = train_models(Xtr_s, y_tr.values, Xcv_s, y_cv.values, cfg)
    xgb = models["xgboost"]                                    # refit on train only

    # --- write artifacts ---
    joblib.dump(xgb, FINAL / "xgboost_final.joblib")
    xgb.get_booster().save_model(str(FINAL / "xgboost_final.json"))
    joblib.dump(scaler, FINAL / "scaler.joblib")
    (FINAL / "feature_list.json").write_text(json.dumps(list(features), indent=2))

    hp = {"selected_model": "xgboost",
          "tuned": tuning["xgboost"]["best_params"],
          "cv_rmse": tuning["xgboost"]["cv_rmse"],
          "fixed": {"random_state": seed, "n_jobs": -1,
                    "tree_method": "hist", "objective": "reg:squarederror"},
          "trained_on": trained_desc,
          "n_train_rows": int(len(X_tr)), "n_features": len(features)}
    (FINAL / "hyperparameters.json").write_text(json.dumps(hp, indent=2))

    files = ["xgboost_final.json", "xgboost_final.joblib", "scaler.joblib",
             "feature_list.json", "hyperparameters.json"]
    lines = [f"{_sha256(FINAL / f)}  {f}" for f in files]
    lines.append(f"{_sha256(Path(parquet))}  ../../{parquet}")
    (FINAL / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    print(f"[freeze] wrote {len(files)} artifacts + SHA256SUMS to {FINAL}/")
    print(f"[freeze] features={len(features)} train_rows={len(X_tr)} "
          f"tuned={hp['tuned']}")
    return hp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/model.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
