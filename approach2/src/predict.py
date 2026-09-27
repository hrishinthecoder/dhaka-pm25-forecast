"""Inference entrypoint for the FROZEN next-day PM2.5 model.

    python -m src.predict --date YYYY-MM-DD

Loads the frozen artifacts in models/final/, builds the day-t feature row for the
requested local date from the processed master, and emits the next-day (t+1)
24-h mean PM2.5 (µg/m³) + US-EPA 2024 AQI category.

Refuses dates whose features are unavailable (date out of range, or any of the
45 model features missing near a coverage gap) with a clear message — the model
never fabricates predictor history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.features.daily import EPA_LABELS, build_daily, to_aqi  # noqa: F401

FINAL = Path("models/final")
DEFAULT_DATA = "data/processed/dhaka_aq_master_singlestation.parquet"
DEFAULT_CONFIG = "config/model.yaml"


class FeatureUnavailable(Exception):
    """Raised when the requested date cannot be turned into a complete feature row."""


def load_artifacts(final_dir: Path = FINAL):
    """Return (ordered_feature_list, scaler, model) from the frozen pack."""
    flist = final_dir / "feature_list.json"
    if not flist.exists():
        raise FileNotFoundError(
            f"{flist} missing — run the freeze step (see docs/MASTER_DOCUMENTATION.md, Section 9).")
    features = json.loads(flist.read_text())
    scaler = joblib.load(final_dir / "scaler.joblib")
    model = joblib.load(final_dir / "xgboost_final.joblib")
    return features, scaler, model


def verify_checksums(final_dir: Path = FINAL) -> tuple[bool, list[tuple[str, str]]]:
    """Check every entry in models/final/SHA256SUMS. Returns (all_ok, [(name, status)]).

    Status is "ok" / "MISMATCH" / "missing". Used by the app to refuse predictions
    when the frozen artifacts have been tampered with.
    """
    sums = final_dir / "SHA256SUMS"
    if not sums.exists():
        return False, [("SHA256SUMS", "missing")]
    results: list[tuple[str, str]] = []
    all_ok = True
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        want, name = line.split("  ", 1)
        p = (final_dir / name).resolve()
        if not p.exists():
            results.append((name, "missing")); all_ok = False; continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        ok = (got == want)
        all_ok = all_ok and ok
        results.append((name, "ok" if ok else "MISMATCH"))
    return all_ok, results


def load_feature_frame(config_path: str = DEFAULT_CONFIG,
                       data_path: str = DEFAULT_DATA):
    """Build the daily feature frame from the (read-only) master. Returns (feat, built)."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    df = pd.read_parquet(data_path)
    return build_daily(df, cfg)


def _predict_from_frame(feat, built, model_features, scaler, model, date: str) -> dict:
    """Core single-date prediction shared by the CLI and the app (one code path)."""
    if set(model_features) - set(built):
        raise FeatureUnavailable(
            "Frozen feature_list does not match the current feature builder: "
            f"{sorted(set(model_features) - set(built))}")

    ts = pd.Timestamp(date)
    if ts not in feat.index:
        lo, hi = feat.index.min().date(), feat.index.max().date()
        raise FeatureUnavailable(
            f"date {date} is outside the available day-t range [{lo} .. {hi}]; "
            "cannot build a feature row.")

    row = feat.loc[ts, model_features]
    if row.isna().any():
        missing = list(row.index[row.isna()])
        raise FeatureUnavailable(
            f"features unavailable for day-t {date} (missing/near a coverage gap): "
            f"{missing}. The model refuses to fabricate predictor history.")

    Xs = scaler.transform(row.to_numpy(dtype=float).reshape(1, -1))
    yhat = float(model.predict(Xs)[0])
    aqi = str(to_aqi(pd.Series([yhat])).iloc[0])
    return {"as_of_date": ts.date().isoformat(),
            "target_date": (ts + pd.Timedelta(days=1)).date().isoformat(),
            "pm25_next_day_ugm3": round(yhat, 2),
            "aqi_category": aqi}


def available_dates(feat, model_features) -> pd.DatetimeIndex:
    """Day-t dates whose full feature row is present (predictable dates)."""
    return feat[model_features].dropna().index


def predict_series(feat, model_features, scaler, model) -> pd.Series:
    """Vectorized next-day predictions for every predictable day-t.

    Returns a Series indexed by **day-t** (the as-of date); the forecast is for t+1.
    Numerically identical to calling `_predict_from_frame` per date.
    """
    X = feat[model_features].dropna()
    if X.empty:
        return pd.Series(dtype=float)
    preds = model.predict(scaler.transform(X.to_numpy(dtype=float)))
    return pd.Series(preds, index=X.index, name="pm25_pred_next_day")


def predict_for_date(date: str, config_path: str = DEFAULT_CONFIG,
                     data_path: str = DEFAULT_DATA, final_dir: Path = FINAL) -> dict:
    """Next-day PM2.5 + AQI for the day-t local `date` (YYYY-MM-DD)."""
    features, scaler, model = load_artifacts(final_dir)
    feat, built = load_feature_frame(config_path, data_path)
    return _predict_from_frame(feat, built, features, scaler, model, date)


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen next-day PM2.5 forecast for Dhaka.")
    ap.add_argument("--date", required=True, help="day-t local date, YYYY-MM-DD")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--data", default=DEFAULT_DATA)
    args = ap.parse_args()
    try:
        out = predict_for_date(args.date, args.config, args.data)
    except (FeatureUnavailable, FileNotFoundError) as exc:
        raise SystemExit(f"[predict] REFUSED: {exc}")
    print(json.dumps(out, indent=2))
    print(f"\nNext-day ({out['target_date']}) PM2.5 = {out['pm25_next_day_ugm3']} "
          f"µg/m³  →  AQI: {out['aqi_category']}")


if __name__ == "__main__":
    main()
