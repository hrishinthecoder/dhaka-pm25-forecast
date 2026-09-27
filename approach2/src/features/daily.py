"""Leakage-safe daily feature engineering for next-day PM2.5 (Option A).

Every produced feature describes information available by **end of day t**. The
target is the **day t+1** daily-mean PM2.5. `feature_availability()` documents the
availability time of every model column to prove forecast-validity.

Leakage rules enforced here:
  * No t+1 fields ever enter the feature matrix.
  * Same-station covariates `openaq_o3` / `openaq_pm10` are EXCLUDED as predictors
    (they come from the forecast target's station and are not available at forecast
    time). See LEAKAGE_EXCLUDED.
  * No satellite PM2.5 (omaq_*/acag_*) as predictors (none present in the rebuilt
    master anyway; guarded by name).

ERA5 is **reanalysis** used as a day-t predictor proxy. An operational deployment
would substitute a numerical-weather-prediction forecast for day t+1; for this
study we follow the standard reanalysis-as-predictor convention and document it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PM25_GROUND_TRUTH = "openaq_pm25"
TIME_COL = "datetime_utc"
LOCAL_TZ = "Asia/Dhaka"

# Same-station covariates excluded as predictors (not available at forecast time).
LEAKAGE_EXCLUDED = ["openaq_o3", "openaq_pm10"]

# ERA5 hourly columns -> daily aggregation strategy.
ERA5_MEAN_VARS = [
    "era5_temperature_2m", "era5_relative_humidity_2m", "era5_dew_point_2m",
    "era5_surface_pressure", "era5_cloud_cover", "era5_wind_speed_10m",
    "era5_boundary_layer_height",
]
ERA5_SUM_VARS = ["era5_precipitation", "era5_shortwave_radiation"]  # accumulations
WIND_DIR = "era5_wind_direction_10m"
WIND_SPD = "era5_wind_speed_10m"
BLH_FLAG = "era5_blh_imputed"
FIRMS_VARS = ["firms_count", "firms_frp_sum"]
CALENDAR_DAY = ["day_of_week", "month", "is_weekend", "is_holiday", "is_ramadan"]

# US EPA 2024 24-h PM2.5 breakpoints (µg/m³).
EPA_BINS = [-0.1, 9.0, 35.4, 55.4, 125.4, 225.4, np.inf]
EPA_LABELS = ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"]


def to_aqi(pm: pd.Series) -> pd.Series:
    return pd.cut(pm, bins=EPA_BINS, labels=EPA_LABELS)


def _local_date(df: pd.DataFrame) -> pd.Series:
    t = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    return t.dt.tz_convert(LOCAL_TZ).dt.normalize().dt.tz_localize(None)


def _daily_pm25_target(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Daily-mean PM2.5 + coverage flags + next-day target + AQI category."""
    min_hours = cfg["coverage"]["min_hourly_obs_per_day"]
    x = df[[TIME_COL, PM25_GROUND_TRUTH]].copy()
    x["date"] = _local_date(df)
    x = x.dropna(subset=["date"])
    g = x.groupby("date")[PM25_GROUND_TRUTH]
    daily = pd.DataFrame({"pm25_mean": g.mean(), "n_hours": g.count()})
    daily = daily.asfreq("D")
    daily.index.name = "date"
    daily["valid"] = daily["n_hours"].fillna(0) >= min_hours
    daily["y_next"] = daily["pm25_mean"].shift(-1)             # day t+1 = target
    daily["valid_next"] = daily["valid"].shift(-1).fillna(False)
    daily["pair_valid"] = daily["valid"] & daily["valid_next"]  # both days clear 18 h
    daily["aqi_next"] = to_aqi(daily["y_next"])
    return daily


def _add_pm25_history(daily: pd.DataFrame, cfg: dict) -> list[str]:
    """PM2.5 lag/rolling features (day t and earlier only). Returns column names."""
    f = cfg["features"]
    cols = ["pm25_t"]
    daily["pm25_t"] = daily["pm25_mean"]                       # day t (available)
    for L in f["pm25_lags"]:
        daily[f"pm25_lag{L}"] = daily["pm25_mean"].shift(L)
        cols.append(f"pm25_lag{L}")
    for W in f["pm25_rolling_windows"]:
        daily[f"pm25_roll_mean{W}"] = daily["pm25_mean"].rolling(W).mean()
        daily[f"pm25_roll_std{W}"] = daily["pm25_mean"].rolling(W).std()
        cols += [f"pm25_roll_mean{W}", f"pm25_roll_std{W}"]
    return cols


def _exogenous_daily(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Daily ERA5 / FIRMS / calendar features (no PM2.5). All are day-t fields."""
    aggs = cfg["features"].get("met_aggregations", ["mean", "min", "max"])
    firms_lags = cfg["features"].get("firms_lags", [0, 1, 2])

    x = df.copy()
    x["date"] = _local_date(df)
    x = x.dropna(subset=["date"])
    g = x.groupby("date")

    out = {}
    cols: list[str] = []

    # ERA5 mean/min/max vars.
    for c in ERA5_MEAN_VARS:
        if c not in x.columns:
            continue
        for a in aggs:
            name = f"{c}_{a}"
            out[name] = getattr(g[c], a)()
            cols.append(name)

    # ERA5 accumulation vars -> daily sum.
    for c in ERA5_SUM_VARS:
        if c not in x.columns:
            continue
        name = f"{c}_sum"
        out[name] = g[c].sum(min_count=1)
        cols.append(name)

    # Wind vector decomposition (handles directional circularity).
    if WIND_DIR in x.columns and WIND_SPD in x.columns:
        rad = np.deg2rad(x[WIND_DIR])
        # Meteorological convention: direction = where wind blows FROM.
        x["_wind_u"] = -x[WIND_SPD] * np.sin(rad)
        x["_wind_v"] = -x[WIND_SPD] * np.cos(rad)
        gw = x.groupby("date")
        out["wind_u_mean"] = gw["_wind_u"].mean()
        out["wind_v_mean"] = gw["_wind_v"].mean()
        cols += ["wind_u_mean", "wind_v_mean"]

    # BLH imputation flag is NOT a model feature: after the CDS patch (PATH A) the
    # H1-2024 gap is filled with native ERA5, so `era5_blh_imputed` is constant 0 and
    # `era5_blh_imputed_day` would be a dead (zero-variance) column. The hourly
    # provenance flag is retained in the parquet but never fed to the models.

    # FIRMS daily (already daily-broadcast to hours -> max == that day's value).
    for c in FIRMS_VARS:
        if c not in x.columns:
            continue
        out[c] = g[c].max()
        cols.append(c)

    # Calendar (day-level constants).
    for c in CALENDAR_DAY:
        if c not in x.columns:
            continue
        out[c] = g[c].max().astype(int)
        cols.append(c)

    exo = pd.DataFrame(out).asfreq("D")
    exo.index.name = "date"

    # FIRMS lags (0,1,2) on the daily series.
    for c in FIRMS_VARS:
        if c not in exo.columns:
            continue
        for L in firms_lags:
            name = f"{c}_lag{L}"
            exo[name] = exo[c].shift(L)
            cols.append(name)
        cols.remove(c)  # use the explicit lag0 column instead of the bare name
    return exo, cols


def build_daily(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Full standalone daily feature frame. Returns (frame, model_feature_cols)."""
    daily = _daily_pm25_target(df, cfg)
    pm_cols = _add_pm25_history(daily, cfg)
    exo, exo_cols = _exogenous_daily(df, cfg)
    feat = daily.join(exo, how="left")
    features = pm_cols + exo_cols
    # Hard leakage guard: no excluded covariate / satellite ever in the matrix.
    bad = [c for c in features
           if c in LEAKAGE_EXCLUDED or "omaq" in c.lower() or "acag" in c.lower()]
    if bad:
        raise ValueError(f"Leakage columns leaked into feature set: {bad}")
    return feat, features


def add_exogenous_features(df: pd.DataFrame, feat: pd.DataFrame,
                           cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Notebook helper: join ERA5/FIRMS/calendar onto an already-built `feat`.

    `feat` is expected to already carry the PM2.5 history columns (Step 4). Returns
    the augmented frame and the FULL ordered model-feature list.
    """
    pm_cols = [c for c in feat.columns if c.startswith("pm25_") and c != "pm25_mean"]
    exo, exo_cols = _exogenous_daily(df, cfg)
    merged = feat.join(exo, how="left")
    return merged, pm_cols + exo_cols


def holdout_start(idx: pd.DatetimeIndex, cfg: dict) -> pd.Timestamp | None:
    """First day of the anchored holdout, or None for the legacy calendar split.

    Anchored mode: the holdout is the LAST `holdout_months` of *available* labeled
    pairs, anchored to the last valid day-t in `idx`. Single-sensor 24434 ends
    2025-03, so a fixed `>=2025` calendar split would leave too few test days; we
    anchor to the data instead. Returns the inclusive holdout start timestamp.
    """
    s = cfg["split"]
    if s.get("mode") == "anchored_holdout":
        return idx.max() - pd.DateOffset(months=s.get("holdout_months", 12))
    return None


def split_masks(idx: pd.DatetimeIndex, cfg: dict):
    """(train, val, test) boolean masks over a day-t DatetimeIndex.

    anchored_holdout: test = last `holdout_months` of available pairs; train = all
    earlier pairs; val = empty (CV tuning and the final refit both use the full
    pre-holdout window). Legacy mode falls back to the fixed train_end/val_year/
    test_start calendar split.
    """
    s = cfg["split"]
    hs = holdout_start(idx, cfg)
    if hs is not None:
        te = idx >= hs
        tr = idx < hs
        va = np.zeros(len(idx), dtype=bool)
        return tr, va, te
    tr = idx <= pd.Timestamp(s["train_end"])
    va = idx.year == s["val_year"]
    te = idx >= pd.Timestamp(s["test_start"])
    return tr, va, te


def model_matrix(feat: pd.DataFrame, features: list[str]
                 ) -> tuple[pd.DataFrame, pd.Series]:
    """Rows where the (t -> t+1) pair is valid AND every feature is present."""
    use = feat[feat["pair_valid"]].copy()
    use = use.dropna(subset=features + ["y_next"])
    return use[features], use["y_next"]


def feature_availability(features: list[str]) -> pd.DataFrame:
    """One row per feature with its availability time (forecast-validity proof)."""
    rows = []
    for c in features:
        if c.startswith("pm25_lag"):
            when, src = f"day t-{c.replace('pm25_lag','')}", "OpenAQ ground PM2.5 history"
        elif c.startswith("pm25_roll"):
            when, src = "day t (window ends at t)", "OpenAQ ground PM2.5 history"
        elif c == "pm25_t":
            when, src = "day t (24-h mean)", "OpenAQ ground PM2.5"
        elif c.startswith("era5_"):
            when, src = "day t", "ERA5 reanalysis (Open-Meteo) — proxy for NWP forecast"
        elif c.startswith("wind_"):
            when, src = "day t", "ERA5 reanalysis wind vector"
        elif c.endswith("_lag0"):
            when, src = "day t", "NASA FIRMS fire activity"
        elif c.endswith(("_lag1", "_lag2")):
            when, src = f"day t-{c[-1]}", "NASA FIRMS fire activity"
        else:
            when, src = "deterministic (known ahead)", "calendar"
        rows.append({"feature": c, "available_by": when, "source": src})
    return pd.DataFrame(rows)
