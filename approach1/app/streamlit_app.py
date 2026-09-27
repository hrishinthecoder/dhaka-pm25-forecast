"""
streamlit_app.py — Phase 9: LOCAL proof-of-concept for the next-day PM2.5 forecast.

WHAT IT SHOWS
    Loads the persisted LightGBM pipeline and the supervised daily table, lets the user pick
    an origin day t (or upload a CSV of trailing features), and displays the predicted
    next-day daily-mean PM2.5 with its US-EPA-2024 AQI category. A recent-window chart
    overlays the model against the persistence baseline and the actual measured value so the
    value-add is visible.

SCOPE (explicit)
    Local proof-of-concept ONLY. It is not deployed, not shared, and wired to no external
    recipient — those are human decisions, not the agent's. Run with:
        streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Make src/ importable so we reuse the exact training-time preprocessing.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from utils import load_config, get_paths           # noqa: E402
from models_tree import predict_bundle             # noqa: E402


def aqi_category(pm25: float, cfg: dict) -> str:
    """Map a PM2.5 concentration to its AQI band using the config breakpoints."""
    for lo, hi, _, _, name in cfg["aqi"]["breakpoints"]:
        if lo <= pm25 <= hi:
            return name
    return "Out of range"


@st.cache_resource
def _load(cfg_path=None):
    """Load config, the supervised table, and the LightGBM bundle (cached across reruns)."""
    cfg = load_config()
    paths = get_paths(cfg)
    table = pd.read_parquet(paths["processed_dir"] / "supervised_daily.parquet")
    feat_info = json.loads((paths["processed_dir"] / "feature_groups.json").read_text())
    bundle = joblib.load(paths["artifacts_dir"] / "model_lightgbm.joblib")
    return cfg, table, feat_info["core_cols"], bundle


def main():
    st.set_page_config(page_title="Dhaka next-day PM2.5", layout="wide")
    st.title("Dhaka next-day PM2.5 forecast — local proof-of-concept")
    st.caption("Single-station target (sensor 24434, US diplomatic-post reference monitor). "
               "ERA5 reanalysis meteorology. LightGBM, walk-forward validated. "
               "Local PoC — not a deployed service.")

    cfg, table, core_cols, bundle = _load()
    preds_all = predict_bundle(bundle, table[core_cols])
    table = table.assign(pred_next_day=preds_all)

    # ---- pick an origin day ----
    days = list(table.index)
    default_i = len(days) - 30
    sel = st.select_slider("Origin day t (forecast is for t+1)",
                           options=days, value=days[default_i],
                           format_func=lambda d: pd.Timestamp(d).date().isoformat())
    row = table.loc[sel]
    pred = float(row["pred_next_day"])
    persist = float(row["pm25_lag_1"])
    actual = float(row["y_next_day"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted next-day PM2.5 (µg/m³)", f"{pred:.1f}", help=aqi_category(pred, cfg))
    c1.write(f"**AQI band:** {aqi_category(pred, cfg)}")
    c2.metric("Persistence baseline (today's mean)", f"{persist:.1f}")
    c3.metric("Actual next-day (measured)", f"{actual:.1f}",
              delta=f"{pred - actual:+.1f} model error")

    # ---- recent-window chart: model vs persistence vs actual ----
    st.subheader("Recent window — model vs persistence vs actual")
    win = st.slider("Window length (days)", 30, 240, 90)
    end = days.index(sel)
    sub = table.iloc[max(0, end - win):end + 1]
    chart = pd.DataFrame({
        "actual (t+1)": sub["y_next_day"].values,
        "model (t+1)": sub["pred_next_day"].values,
        "persistence (t+1)": sub["pm25_lag_1"].values,
    }, index=[pd.Timestamp(d).date() for d in sub.index])
    st.line_chart(chart)

    # ---- optional CSV upload of trailing core features ----
    st.subheader("Or upload a CSV of trailing core features")
    st.caption(f"CSV must contain these columns: {', '.join(core_cols[:6])} … "
               f"({len(core_cols)} total).")
    up = st.file_uploader("Upload feature CSV", type="csv")
    if up is not None:
        user_df = pd.read_csv(up)
        missing = [c for c in core_cols if c not in user_df.columns]
        if missing:
            st.error(f"Missing {len(missing)} required columns, e.g. {missing[:5]}")
        else:
            yhat = predict_bundle(bundle, user_df[core_cols])
            user_df["predicted_next_day_pm25"] = yhat
            user_df["aqi_band"] = [aqi_category(v, cfg) for v in yhat]
            st.dataframe(user_df[["predicted_next_day_pm25", "aqi_band"]])


if __name__ == "__main__":
    main()
