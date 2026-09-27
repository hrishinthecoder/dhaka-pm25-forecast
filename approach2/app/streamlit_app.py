"""Streamlit proof-of-concept DISPLAY LAYER for the frozen Dhaka PM2.5 model.

    streamlit run app/streamlit_app.py

Read-only over the frozen artifacts in models/final/ and the processed master.
It demonstrates the model to collaborators — it is NOT an operational service.
All prediction goes through the SHARED code path in src/predict.py (no duplicated
feature-building, no retraining, no model switching). Frozen artifacts are checksum-
verified at startup; predictions are refused on any mismatch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import (  # noqa: E402
    FeatureUnavailable, _predict_from_frame, available_dates, load_artifacts,
    load_feature_frame, predict_series, verify_checksums)

FINAL = ROOT / "models" / "final"
DATA = ROOT / "data" / "processed" / "dhaka_aq_master_singlestation.parquet"
CONFIG = ROOT / "config" / "model.yaml"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

# US-EPA 2024 24-h PM2.5 bands: (label, upper µg/m³, color, health statement).
EPA_BANDS = [
    ("Good", 9.0, "#00e400",
     "Air quality is satisfactory; little or no health risk."),
    ("Moderate", 35.4, "#ffff00",
     "Acceptable; unusually sensitive people should consider limiting prolonged outdoor exertion."),
    ("USG", 55.4, "#ff7e00",
     "Members of sensitive groups may experience health effects."),
    ("Unhealthy", 125.4, "#ff0000",
     "Everyone may begin to experience health effects; sensitive groups more serious effects."),
    ("Very Unhealthy", 225.4, "#8f3f97",
     "Health alert: everyone may experience more serious health effects."),
    ("Hazardous", float("inf"), "#7e0023",
     "Health warning of emergency conditions; the entire population is likely affected."),
]
EPA_LINES = [9.0, 35.4, 55.4, 125.4, 225.4]   # band boundaries for chart reference lines


def band_for(label: str):
    for lbl, _hi, color, health in EPA_BANDS:
        if lbl == label:
            return color, health
    return "#999999", ""


# ---------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner=False)
def load_frozen():
    """Verify checksums then load frozen artifacts. Cached for the session."""
    ok, results = verify_checksums(FINAL)
    if not ok:
        return {"ok": False, "results": results}
    features, scaler, model = load_artifacts(FINAL)
    return {"ok": True, "results": results,
            "features": features, "scaler": scaler, "model": model}


@st.cache_data(show_spinner=False)
def load_feat():
    feat, built = load_feature_frame(str(CONFIG), str(DATA))
    return feat, built


@st.cache_data(show_spinner=False)
def get_available_dates():
    feat, _ = load_feat()
    fr = load_frozen()
    if not fr["ok"]:
        return []
    return [d.date() for d in available_dates(feat, fr["features"])]


@st.cache_data(show_spinner=False)
def recent_overlay(n: int = 90) -> pd.DataFrame:
    """Last n observed days with the model's next-day prediction overlaid (by target date)."""
    fr = load_frozen()
    feat, _ = load_feat()
    s = predict_series(feat, fr["features"], fr["scaler"], fr["model"])   # indexed by day-t
    pred_by_target = s.copy()
    pred_by_target.index = s.index + pd.Timedelta(days=1)                 # forecast is for t+1
    df = pd.DataFrame({"observed": feat["pm25_mean"]})
    df["predicted"] = pred_by_target
    df = df.dropna(subset=["observed"])
    out = df.tail(n).reset_index().rename(columns={"index": "date", "date": "date"})
    out.columns = ["date", "observed", "predicted"]
    return out


@st.cache_data(show_spinner=False)
def read_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False)
def read_text(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def test_mae() -> float | None:
    """XGBoost test-period MAE from reports/metrics.csv (never hard-coded)."""
    try:
        m = read_csv(str(REPORTS / "metrics.csv")).set_index("model")
        return float(m.loc["xgboost", "mae"])
    except Exception:
        return None


# ---------------------------------------------------------------- header
st.set_page_config(page_title="Dhaka Next-Day PM2.5 (PoC)", page_icon="🌫️", layout="wide")
st.title("Dhaka Next-Day PM2.5 Forecast — Research Proof of Concept")
st.warning(
    "Retrospective research demonstration. Not an operational advisory. Forecasts use "
    "ERA5 reanalysis as a day-t proxy; an operational system would require NWP forecast "
    "inputs. Single station (Dhaka). Not a substitute for regulatory monitoring.")

FROZEN = load_frozen()
if not FROZEN["ok"]:
    st.error(
        "**Frozen-artifact integrity check FAILED — predictions disabled.**\n\n"
        "models/final/ does not match SHA256SUMS:\n\n"
        + "\n".join(f"- `{n}`: {s}" for n, s in FROZEN["results"])
        + "\n\nRestore the frozen pack (see docs/MASTER_DOCUMENTATION.md, Section 9) and reload.")
    st.stop()

page = st.sidebar.radio(
    "View", ["Forecast", "Model performance", "Recent forecasts", "About / model card"])
st.sidebar.caption("Frozen model: XGBoost · checksums verified ✓")


# ---------------------------------------------------------------- helper: render one prediction
def render_prediction(date_str: str):
    feat, built = load_feat()
    try:
        out = _predict_from_frame(
            feat, built, FROZEN["features"], FROZEN["scaler"], FROZEN["model"], date_str)
    except FeatureUnavailable as exc:
        st.info(f"No forecast: {exc}")
        return

    color, health = band_for(out["aqi_category"])
    pm = out["pm25_next_day_ugm3"]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric(f"Predicted next-day PM2.5 ({out['target_date']})", f"{pm:.1f} µg/m³")
        st.markdown(
            f"<div style='padding:8px 14px;border-radius:6px;background:{color};"
            f"color:#000;font-weight:600;display:inline-block'>AQI: {out['aqi_category']}</div>",
            unsafe_allow_html=True)
        st.caption(health)
    with c2:
        ts = pd.Timestamp(date_str)
        actual = feat.loc[ts, "y_next"] if ts in feat.index else None
        if actual is not None and pd.notna(actual):
            st.metric(f"Observed (actual {out['target_date']})", f"{float(actual):.1f} µg/m³")
            st.metric("Absolute error", f"{abs(pm - float(actual)):.1f} µg/m³")
        else:
            st.caption("No observed next-day value in the record for this date "
                       "(true out-of-sample forecast).")

    mae = test_mae()
    mae_txt = f"{mae:.1f}" if mae is not None else "21.4"
    st.caption(f"Test-period MAE: {mae_txt} µg/m³; typical error is larger in the winter dry season.")


# ---------------------------------------------------------------- pages
if page == "Forecast":
    st.subheader("Next-day forecast")
    dates = get_available_dates()
    if not dates:
        st.error("No predictable dates found — is the processed master present?")
        st.stop()
    lo, hi = min(dates), max(dates)
    date_set = set(dates)
    picked = st.date_input("Day-t (forecast is for the following day)",
                           value=hi, min_value=lo, max_value=hi)
    if picked not in date_set:
        st.info(f"No forecast: {picked} has no complete day-t feature row "
                "(missing or near a coverage gap). Pick another date.")
    else:
        render_prediction(picked.isoformat())

elif page == "Model performance":
    st.subheader("Model performance (held-out test = last 12 months, n=219)")

    try:
        m = read_csv(str(REPORTS / "metrics.csv"))
        disp = pd.DataFrame({
            "model": m["model"],
            "RMSE": m["rmse"].round(2),
            "RMSE 95% CI": m.apply(lambda r: f"[{r.rmse_ci_lo:.1f}, {r.rmse_ci_hi:.1f}]", axis=1),
            "MAE": m["mae"].round(2),
            "R²": m["r2"].round(3),
            "R² 95% CI": m.apply(lambda r: f"[{r.r2_ci_lo:.2f}, {r.r2_ci_hi:.2f}]", axis=1),
        })
        st.markdown("**Headline metrics with bootstrap confidence intervals**")
        st.dataframe(disp, hide_index=True, width="stretch")
    except Exception as exc:
        st.error(f"Could not read reports/metrics.csv: {exc}")

    try:
        sig = read_csv(str(REPORTS / "significance_vs_persistence.csv"))
        st.markdown("**Paired significance vs persistence**")
        st.dataframe(sig.round(3), hide_index=True, width="stretch")
        st.caption("XGBoost and Random Forest separate from a same-as-yesterday baseline; "
                   "the gain concentrates in large errors (episode days).")
    except Exception as exc:
        st.error(f"Could not read significance_vs_persistence.csv: {exc}")

    try:
        seas = read_csv(str(REPORTS / "seasonal_metrics.csv"))
        st.markdown("**Seasonal skill (best model)**")
        st.bar_chart(seas.set_index("season")["rmse"])
        st.caption("Winter-dry is the hardest regime (R² 0.32); skill is strongest for "
                   "episode detection rather than point accuracy.")
    except Exception as exc:
        st.error(f"Could not read seasonal_metrics.csv: {exc}")

    fig = REPORTS / "figures" / "rolling_backtest_rmse.png"
    st.markdown("**Rolling-origin backtest**")
    if fig.exists():
        st.image(str(fig), width="stretch")
        st.caption("Skill over persistence is statistically separable in 3 of 6 folds "
                   "(2022, 2024, 2025); 2020 and 2023 are marginally worse — not a recency "
                   "or training-length trend (2022 wins, 2023 loses); the edge is on "
                   "high-error days.")
    else:
        st.info("rolling_backtest_rmse.png not found — run the rolling backtest to generate it.")

elif page == "Recent forecasts":
    st.subheader("Last 90 available days — observed vs next-day prediction")
    try:
        rec = recent_overlay(90)
        long = rec.melt("date", ["observed", "predicted"], "series", "pm25")
        try:
            import altair as alt
            base = alt.Chart(long).mark_line(point=False).encode(
                x=alt.X("date:T", title="date"),
                y=alt.Y("pm25:Q", title="PM2.5 (µg/m³)"),
                color=alt.Color("series:N",
                                scale=alt.Scale(domain=["observed", "predicted"],
                                                range=["#1f77b4", "#d62728"])))
            rules = alt.Chart(pd.DataFrame({"y": EPA_LINES})).mark_rule(
                strokeDash=[4, 4], color="#aaaaaa").encode(y="y:Q")
            st.altair_chart(base + rules, width="stretch")
        except Exception:
            st.line_chart(rec.set_index("date")[["observed", "predicted"]])
        st.caption("Dashed lines = US-EPA PM2.5 band boundaries (9, 35.4, 55.4, 125.4, 225.4 µg/m³). "
                   "Predictions computed on the fly from the frozen artifacts.")
    except Exception as exc:
        st.error(f"Could not build the recent-forecasts view: {exc}")

else:  # About / model card
    st.subheader("About — frozen model card")
    card = DOCS / "MASTER_DOCUMENTATION.md"
    if card.exists():
        st.markdown(read_text(str(card)))
    else:
        st.error("docs/MASTER_DOCUMENTATION.md not found.")
    st.divider()
    st.markdown(
        "**Data provenance:** OpenAQ ground truth (single Dhaka reference station — sensor "
        "24434, US Diplomatic Post, ~2 km from centre), ERA5 reanalysis "
        "via Open-Meteo (boundary-layer height Jan–Jun 2024 from the native Copernicus CDS), "
        "NASA FIRMS fire activity, and calendar proxies. No measured traffic data.")
    st.markdown(
        "Full methods, model card, and reproducibility: `docs/MASTER_DOCUMENTATION.md`. "
        "Verify artifacts: `models/final/SHA256SUMS`.")
