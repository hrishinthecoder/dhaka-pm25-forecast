"""
features.py — leakage-safe DAILY feature engineering for the next-day forecast.

GUIDING RULE (the leakage firewall)
    To predict the daily mean of day t+1, a feature may use information available only
    THROUGH day t. The single deliberate exception is CALENDAR features of the target day
    t+1 (month, weekday, holiday) — these are deterministic and genuinely known in advance
    (we always know tomorrow's date), so using them leaks nothing.

WHAT WE BUILD (all indexed by origin day t)
    - ERA5 meteorology: daily mean of day t for each core variable, plus a 3-day trailing
      mean (recent weather context). [group: era5_meteorology]
    - Calendar/cyclical: sin/cos of month, weekday, day-of-year for the TARGET day t+1,
      plus weekend and Bangladesh-holiday flags. [group: calendar_cyclical]
    - PM2.5 autoregressive lags: value on days t, t-1, t-2, t-6 (i.e. 1/2/3/7 days before
      the target). lag-1 is exactly the persistence prediction, used here as a feature.
      [group: pm25_lags]
    - PM2.5 rolling stats: mean/std/min/max over trailing 3/7/14-day windows ending at day
      t (the target day t+1 is naturally excluded). [group: pm25_rolling]
    - FIRMS fire: daily detection count and total fire radiative power on day t; frp_sum is
      filled with 0 when count==0 (no fire => no radiative power). [group: fire]

No model is fit here and no statistics are learned across folds — these are deterministic,
per-timestamp transforms, so they are leakage-safe to compute once on the full series.
Learned steps (scalers, IsolationForest, imputers) happen fold-internally in Phase 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Mapping kept so the feature-dictionary report can tag each feature with its ablation
# group and its temporal availability. Filled as features are built.
FEATURE_GROUPS: dict[str, str] = {}


def _tag(cols, group):
    """Record each feature's ablation group in FEATURE_GROUPS (used by the report)."""
    for c in cols:
        FEATURE_GROUPS[c] = group


def _daily_aggregate_master(master: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Aggregate the hourly master to one row per LOCAL day (Asia/Dhaka).

    Returns a daily frame holding ERA5 daily means, FIRMS daily sums, the ACAG daily
    mean, and a boundary-layer-height missingness flag. Indexed by local_day.
    """
    tz = cfg["data"]["local_timezone"]
    local = master["datetime_utc"].dt.tz_convert(tz)
    day = local.dt.normalize().dt.tz_localize(None)
    g = master.assign(local_day=day).groupby("local_day")

    era5_cols = cfg["features"]["era5_meteorology"]
    daily = g[era5_cols].mean()  # daily mean of each met variable over day t

    # Missingness flag for boundary layer height (the only met col with real gaps, ~1.75%).
    # A day is flagged if it did not have a full 24 valid BLH hours.
    blh = "era5_boundary_layer_height"
    if blh in master.columns:
        valid_blh = g[blh].count()
        daily[blh + "_was_missing"] = (valid_blh < 24).astype(int)

    # FIRMS: sum detections / radiative power over the day. frp_sum is mostly NaN
    # (no fire) -> fill 0 per config so "no fire" reads as zero power, not missing.
    fill = cfg["preprocessing"].get("firms_frp_sum_fill", 0)
    if "firms_count" in master.columns:
        daily["firms_count"] = g["firms_count"].sum(min_count=1)
    if "firms_frp_sum" in master.columns:
        daily["firms_frp_sum"] = g["firms_frp_sum"].sum(min_count=1)
        # Where the day had detections but frp is NaN, or no fire at all -> fill.
        cnt = daily.get("firms_count")
        if cnt is not None:
            daily.loc[cnt.fillna(0) == 0, "firms_frp_sum"] = fill
        daily["firms_frp_sum"] = daily["firms_frp_sum"].fillna(fill)

    # ACAG coarse satellite anchor (near-static annual/monthly grid) -> daily mean.
    if "acag_pm25" in master.columns:
        daily["acag_pm25"] = g["acag_pm25"].mean()

    # Ablation-only CAMS model fields (daily means of day t). NOT in the core set; built
    # so Phase 8 can toggle them. Includes the near-circular omaq_pm2_5 (CAMS PM2.5 model).
    abl_cols = ["omaq_carbon_monoxide", "omaq_nitrogen_dioxide", "omaq_sulphur_dioxide",
                "omaq_ozone", "omaq_dust", "omaq_aerosol_optical_depth", "omaq_pm2_5"]
    for c in abl_cols:
        if c in master.columns:
            daily[c] = g[c].mean()

    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "local_day"
    return daily


def build_features(master: pd.DataFrame, daily_target: pd.DataFrame,
                   cfg: dict) -> pd.DataFrame:
    """Assemble the full daily feature matrix indexed by origin day t.

    Args:
        master: hourly merged master (for ERA5/FIRMS/ACAG features).
        daily_target: daily single-station PM2.5 (`daily_mean_target` output) — the source
            of the autoregressive PM2.5 lag/rolling features.
        cfg: config dict (feature lists, lag days, rolling windows, holiday country).

    Returns:
        DataFrame indexed by origin day t, one column per feature. Rows span the union of
        the daily calendar; the build script later keeps only rows that have a valid
        next-day target. Trees tolerate the residual NaNs; a fold-internal imputer in
        Phase 5 fills them on train-only statistics.
    """
    FEATURE_GROUPS.clear()

    # ---- ERA5 / FIRMS / ACAG daily aggregates (knowable on day t) ----
    daily_master = _daily_aggregate_master(master, cfg)

    # Continuous daily calendar so lag/rolling shifts are by real days, not row position.
    full_idx = pd.date_range(daily_master.index.min(),
                             max(daily_master.index.max(), daily_target.index.max()),
                             freq="D")
    feats = pd.DataFrame(index=full_idx)
    feats.index.name = "origin_day"

    # ERA5 day-t daily means + 3-day trailing means (recent weather context).
    era5_cols = cfg["features"]["era5_meteorology"]
    dm = daily_master.reindex(full_idx)
    for c in era5_cols:
        feats[c + "_d0"] = dm[c]                                   # day t value
        feats[c + "_roll3"] = dm[c].rolling(3, min_periods=2).mean()  # last 3 days
    _tag([c + "_d0" for c in era5_cols] + [c + "_roll3" for c in era5_cols],
         "era5_meteorology")
    blh_flag = "era5_boundary_layer_height_was_missing"
    if blh_flag in daily_master.columns:
        feats[blh_flag] = dm[blh_flag].fillna(1).astype(int)
        _tag([blh_flag], "era5_meteorology")

    # FIRMS fire (day t).
    for c in ["firms_count", "firms_frp_sum"]:
        if c in daily_master.columns:
            feats[c] = dm[c]
            _tag([c], "fire")

    # ACAG coarse anchor (day t) — kept as a low-frequency anchor only.
    if "acag_pm25" in daily_master.columns:
        feats["acag_pm25"] = dm["acag_pm25"]
        _tag(["acag_pm25"], "pm_adjacent_model_products")  # not in core; available for ablation

    # Ablation-only CAMS model fields (NOT in core). Grouped to match config.features.* so
    # Phase 8 can switch them on. omaq_pm2_5 is the near-circular CAMS PM2.5 model output.
    abl_map = {
        "cams_chemistry": ["omaq_carbon_monoxide", "omaq_nitrogen_dioxide",
                           "omaq_sulphur_dioxide", "omaq_ozone"],
        "pm_adjacent_model_products": ["omaq_dust", "omaq_aerosol_optical_depth"],
        "cams_pm25_benchmark": ["omaq_pm2_5"],
    }
    for group, cols in abl_map.items():
        present = [c for c in cols if c in daily_master.columns]
        for c in present:
            feats[c] = dm[c]
        _tag(present, group)

    # ---- PM2.5 autoregressive features from the single-station daily series ----
    # Reindex the (qualifying-days-only) target mean onto the continuous calendar so
    # shifts move by exactly one day. Missing days stay NaN (trees handle it).
    pm = daily_target["pm25_daily_mean"].reindex(full_idx)

    # Lags: lag_k = value k days before the TARGET day t+1 == pm at day (t-(k-1)).
    # lag_1 == pm(t) == the persistence prediction expressed as a feature.
    lag_days = cfg["features"]["pm25_lags"]["lag_days"]  # [1,2,3,7]
    lag_cols = []
    for k in lag_days:
        col = f"pm25_lag_{k}"
        feats[col] = pm.shift(k - 1)   # k=1 -> shift 0 (today); k=7 -> shift 6 (t-6)
        lag_cols.append(col)
    _tag(lag_cols, "pm25_lags")

    # Rolling stats over trailing windows ENDING at day t (target day excluded by design).
    windows = cfg["features"]["pm25_rolling"]["windows_days"]   # [3,7,14]
    stats = cfg["features"]["pm25_rolling"]["stats"]            # [mean,std,min,max]
    roll_cols = []
    for w in windows:
        r = pm.rolling(w, min_periods=max(2, w // 2))
        for stat in stats:
            col = f"pm25_roll{w}_{stat}"
            feats[col] = getattr(r, stat)()
            roll_cols.append(col)
    _tag(roll_cols, "pm25_rolling")

    # ---- Calendar / cyclical of the TARGET day t+1 (deterministic, known in advance) ----
    target_day = full_idx + pd.Timedelta(days=1)
    cal = pd.DataFrame(index=full_idx)

    def _cyc(name, values, period):
        """Add sin/cos columns encoding a cyclical calendar field (so e.g. Dec≈Jan)."""
        cal[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        cal[f"{name}_cos"] = np.cos(2 * np.pi * values / period)

    _cyc("month", target_day.month, 12)
    _cyc("day_of_week", target_day.dayofweek, 7)
    _cyc("day_of_year", target_day.dayofyear, 366)
    cal["is_weekend"] = target_day.dayofweek.isin([4, 5]).astype(int)  # Fri/Sat in Bangladesh

    # Bangladesh public-holiday flag for the target day (if the holidays package is present).
    try:
        import holidays
        country = cfg["features"]["calendar_cyclical"]["holiday_country"]
        yrs = range(target_day.year.min(), target_day.year.max() + 1)
        bd = holidays.country_holidays(country, years=list(yrs))
        cal["is_public_holiday"] = [int(d.date() in bd) for d in target_day]
    except Exception:
        cal["is_public_holiday"] = 0

    for c in cal.columns:
        feats[c] = cal[c].values
    _tag(list(cal.columns), "calendar_cyclical")

    return feats
