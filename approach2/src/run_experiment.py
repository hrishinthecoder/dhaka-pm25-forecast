"""End-to-end Option-A modeling pipeline (headless, reproducible).

    python -m src.run_experiment --config config/model.yaml

Builds leakage-safe daily features, trains baselines + RF/XGB/LightGBM (time-series
CV tuning, scalers fit on train only), evaluates on the held-out test window with
bootstrap CIs + seasonal breakdown + residual diagnostics, runs SHAP + permutation
importance for the best model, and writes every artifact under reports/ and models/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.features.daily import (EPA_LABELS, build_daily, feature_availability,
                                holdout_start, model_matrix, split_masks, to_aqi)
from src.models.train import (apply_scaler, baselines, fit_scaler, train_models)
from src.evaluate.metrics import (bootstrap_ci, classification_metrics,
                                  reg_metrics, seasonal_metrics,
                                  significance_vs_persistence)
from src.evaluate.interpret import permutation_importances, shap_analysis
from src.evaluate.plots import (confusion_matrix_plot, pred_vs_actual,
                                residual_acf_plot, residual_vs_features)

DEFAULT_DATA = "data/processed/dhaka_aq_master_singlestation.parquet"


def run(config_path: str, data_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = cfg["artifacts"]["random_seed"]
    np.random.seed(seed)
    reports = Path(cfg["artifacts"]["reports_dir"]); reports.mkdir(parents=True, exist_ok=True)
    figs = reports / "figures"; figs.mkdir(parents=True, exist_ok=True)
    models_dir = Path(cfg["artifacts"]["models_dir"]); models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    feat, features = build_daily(df, cfg)
    feature_availability(features).to_csv(reports / "feature_availability.csv", index=False)

    X, y = model_matrix(feat, features)
    tr_m, va_m, te_m = split_masks(X.index, cfg)
    X_tr, y_tr = X[tr_m], y[tr_m]
    X_va, y_va = X[va_m], y[va_m]
    X_te, y_te = X[te_m], y[te_m]
    X_cv = pd.concat([X_tr, X_va]).sort_index()
    y_cv = pd.concat([y_tr, y_va]).sort_index()
    print(f"features={len(features)} | train={len(X_tr)} val={len(X_va)} test={len(X_te)}")

    # Paired daily frame (for baselines that need pm25_t / day-of-year climatology).
    # Boundary = the same anchored holdout start (legacy: train_end / test_start).
    paired = feat[feat["pair_valid"]].dropna(subset=features + ["y_next"])
    hs = holdout_start(X.index, cfg)
    if hs is not None:
        p_tr = paired[paired.index < hs]
        p_te = paired[paired.index >= hs]
    else:
        p_tr = paired[paired.index <= pd.Timestamp(cfg["split"]["train_end"])]
        p_te = paired[paired.index >= pd.Timestamp(cfg["split"]["test_start"])]

    scaler = fit_scaler(X_tr)
    Xtr_s = apply_scaler(scaler, X_tr)
    Xcv_s = apply_scaler(scaler, X_cv)
    Xte_s = apply_scaler(scaler, X_te)
    joblib.dump(scaler, models_dir / "scaler.joblib")

    # --- predictions: baselines + ML ---
    preds = baselines(p_tr, p_te)                       # persistence, climatology
    models, tuning = train_models(Xtr_s, y_tr.values, Xcv_s, y_cv.values, cfg)
    for name, mdl in models.items():
        preds[name] = mdl.predict(Xte_s)

    return evaluate_and_save(models, tuning, preds, Xtr_s, Xte_s,
                             y_tr, y_te, X_te, features, cfg)


def evaluate_and_save(models, tuning, preds, Xtr_s, Xte_s, y_tr, y_te, X_te,
                      features, cfg) -> dict:
    """Score on test (point + bootstrap CI), seasonal split, residual diagnostics,
    AQI classification, SHAP, permutation importance; persist every artifact.

    Shared by the CLI and notebook Step 7 so both emit identical outputs.
    """
    seed = cfg["artifacts"]["random_seed"]
    reports = Path(cfg["artifacts"]["reports_dir"]); reports.mkdir(parents=True, exist_ok=True)
    figs = reports / "figures"; figs.mkdir(parents=True, exist_ok=True)
    models_dir = Path(cfg["artifacts"]["models_dir"]); models_dir.mkdir(parents=True, exist_ok=True)

    # --- metric table (point + 95% bootstrap CI) ---
    rows = []
    for name, yhat in preds.items():
        m = reg_metrics(y_te.values, yhat)
        ci = bootstrap_ci(y_te.values, yhat, n_boot=1000, seed=seed)
        rows.append({"model": name, **m, **ci,
                     "is_baseline": name in ("persistence", "climatology")})
    metrics = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    metrics.to_csv(reports / "metrics.csv", index=False)

    # Pre-registered selection rule: lowest TimeSeriesSplit CV-RMSE among the TUNED
    # models. Selecting on the held-out test RMSE would be selecting on the test set;
    # RandomForest is a fixed baseline with no CV score, so the tuned XGB/LGBM compete
    # for primary. Falls back to test-RMSE order only if no tuned model exists.
    tuned_cv = {m: tuning[m]["cv_rmse"] for m in models
                if m in tuning and "cv_rmse" in tuning[m]}
    best = (min(tuned_cv, key=tuned_cv.get) if tuned_cv
            else metrics[metrics["model"].isin(list(models))].iloc[0]["model"])
    print(f"best ML model (lowest CV-RMSE among tuned): {best}")
    json.dump(tuning, open(reports / "tuning_log.json", "w"), indent=2)

    # --- paired significance vs persistence (CIs cannot, paired tests can) ---
    sig = significance_vs_persistence(y_te.values, preds, baseline="persistence",
                                      n_boot=1000, seed=seed)
    sig.to_csv(reports / "significance_vs_persistence.csv", index=False)
    print("significance vs persistence:")
    print(sig.to_string(index=False))

    seasonal_metrics(X_te.index, y_te.values, preds[best]).to_csv(
        reports / "seasonal_metrics.csv")

    resid = y_te.values - preds[best]
    pred_vs_actual(y_te.values, preds[best], figs, best)
    residual_acf_plot(pd.Series(resid, index=X_te.index), figs, best)
    top_feats = permutation_importances(models[best], Xtr_s, y_tr.values,
                                        Xte_s, y_te.values, features, seed)
    top_feats.to_csv(reports / "permutation_importance.csv", index=False)
    residual_vs_features(resid, X_te, list(top_feats["feature"].head(6)), figs, best)

    aqi_true = to_aqi(pd.Series(y_te.values))
    aqi_pred = to_aqi(pd.Series(preds[best]))
    cls = classification_metrics(aqi_true, aqi_pred, EPA_LABELS)
    confusion_matrix_plot(cls["confusion_matrix"], EPA_LABELS, figs, best)
    json.dump({"accuracy": cls["accuracy"], "macro_f1": cls["macro_f1"], "n": cls["n"]},
              open(reports / "classification_metrics.json", "w"), indent=2)

    try:
        shap_analysis(models[best], Xte_s, features, figs, best).to_csv(
            reports / "shap_importance.csv")
    except Exception as exc:                                   # noqa: BLE001
        print(f"SHAP skipped: {exc}")

    _save_models(models, models_dir)
    _write_model_cards(metrics, tuning, features, cls, best, reports, cfg)

    print("DONE. Artifacts in reports/ and models/.")
    return {"metrics": metrics, "best": best, "classification": cls,
            "permutation_importance": top_feats, "significance": sig, "seasonal":
            seasonal_metrics(X_te.index, y_te.values, preds[best])}


def _save_models(models: dict, models_dir: Path) -> None:
    for name, mdl in models.items():
        joblib.dump(mdl, models_dir / f"{name}.joblib")
    if "xgboost" in models:
        models["xgboost"].get_booster().save_model(str(models_dir / "xgboost.json"))
    if "lightgbm" in models:
        models["lightgbm"].booster_.save_model(str(models_dir / "lightgbm.txt"))


def _write_model_cards(metrics, tuning, features, cls, best, reports, cfg) -> None:
    for _, r in metrics.iterrows():
        name = r["model"]
        kind = "baseline" if r["is_baseline"] else "ML model"
        lines = [
            f"# Model card — {name}", "",
            f"**Type:** {kind}  ", f"**Task:** next-day (t+1) 24-h mean PM2.5 (µg/m³), Option A  ",
            f"**Best model:** {'YES' if name == best else 'no'}  ",
            f"**Seed:** {cfg['artifacts']['random_seed']}", "",
            "## Test-set performance (point + 95% bootstrap CI, n_boot=1000)", "",
            "| metric | value | 95% CI |", "|---|---|---|",
            f"| RMSE | {r['rmse']:.3f} | [{r['rmse_ci_lo']:.3f}, {r['rmse_ci_hi']:.3f}] |",
            f"| MAE | {r['mae']:.3f} | [{r['mae_ci_lo']:.3f}, {r['mae_ci_hi']:.3f}] |",
            f"| R² | {r['r2']:.3f} | [{r['r2_ci_lo']:.3f}, {r['r2_ci_hi']:.3f}] |",
            f"| bias | {r['bias']:.3f} | [{r['bias_ci_lo']:.3f}, {r['bias_ci_hi']:.3f}] |",
            f"| n (test) | {int(r['n'])} | — |", "",
        ]
        if name in tuning:
            lines += ["## Tuning (TimeSeriesSplit n=5 on train+val; refit on train)", "",
                      f"```json\n{json.dumps(tuning[name], indent=2)}\n```", ""]
        if name == best:
            lines += ["## AQI classification (US EPA 2024)", "",
                      f"- accuracy: {cls['accuracy']:.3f}",
                      f"- macro-F1: {cls['macro_f1']:.3f}", ""]
        lines += ["## Features", "",
                  f"{len(features)} day-t features (no t+1, no same-station "
                  "o3/pm10, no satellite PM2.5). See reports/feature_availability.csv.", "",
                  "## Notes",
                  "- ERA5 = reanalysis used as a day-t predictor proxy.",
                  "- BLH for the 2024 H1 gap is sourced from native ERA5 (Copernicus "
                  "CDS); `era5_blh_imputed_day` is now constant 0 (no imputed hours "
                  "remain). Fallback when CDS is unavailable: train-years-only "
                  "(<=2023) month×hour climatology.", ""]
        (reports / f"model_card_{name}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/model.yaml")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data = args.data or cfg.get("data", {}).get("input_path") or DEFAULT_DATA
    run(args.config, data)


if __name__ == "__main__":
    main()
