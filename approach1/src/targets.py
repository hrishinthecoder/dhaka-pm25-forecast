"""
targets.py — build the daily single-station target and the next-day label.

WHAT THE TARGET IS (state this verbatim in the manuscript)
    `pm25_next_day_daily_mean` = the daily-MEAN PM2.5 of sensor 24434 on day t+1,
    predicted from information available through day t. The daily mean is taken over the
    LOCAL calendar day (Asia/Dhaka, UTC+6) and a day counts only if it has >= 18 valid
    hourly observations. It is NOT a daily AQI and NOT a same-hour +24h shift.

WHY THESE RULES
    - Local day: "next day" must mean a Dhaka citizen's day, not a UTC day.
    - >=18 hours: a daily mean from a handful of hours is unreliable; the threshold keeps
      only well-sampled days (Phase-0 verified this yields ~1,899 days / 1,746 pairs).
    - t+1 from t: the one-day-ahead horizon is the whole point of the forecast.
"""

from __future__ import annotations

import pandas as pd


def daily_mean_target(hourly_pm25: pd.Series, cfg: dict) -> pd.DataFrame:
    """Collapse the hourly single-station series to a clean daily-mean series.

    Args:
        hourly_pm25: hourly PM2.5 (UTC-indexed) from `data_loader.load_target_sensor`.
        cfg: config dict (uses target.aggregate_on, target.min_hours_per_day, local tz).

    Returns:
        DataFrame indexed by local calendar day (`local_day`, a tz-naive date) with:
            - `pm25_daily_mean`: mean PM2.5 over that local day,
            - `n_valid_hours`: how many valid hourly obs fed the mean.
        Only days meeting the >=min_hours_per_day rule are kept.

    Why it matters: this daily series is both the prediction target and the source of the
    autoregressive PM2.5 lag/rolling features. Getting the day boundary and coverage rule
    right here is what makes the whole supervised problem well-posed.
    """
    tz = cfg["data"]["local_timezone"]
    min_hours = cfg["target"]["min_hours_per_day"]

    # Convert each UTC timestamp to Dhaka local time, then take the local calendar day.
    local = hourly_pm25.index.tz_convert(tz)
    local_day = pd.Series(local.normalize().tz_localize(None), index=hourly_pm25.index)

    grouped = hourly_pm25.groupby(local_day.values)
    daily = pd.DataFrame({
        "pm25_daily_mean": grouped.mean(),
        "n_valid_hours": grouped.size(),
    })
    daily.index.name = "local_day"
    daily.index = pd.to_datetime(daily.index)

    # Keep only well-sampled days (the >=18-hour rule).
    daily = daily[daily["n_valid_hours"] >= min_hours].sort_index()
    return daily


def build_supervised_target(daily: pd.DataFrame) -> pd.DataFrame:
    """Attach the next-day (t+1) label to each origin day t.

    For an origin day t, the label is the daily mean of day t+1 — but ONLY when t+1 is
    the literal next calendar day AND itself a qualifying day. Pairs where t+1 is missing
    (a gap day, or fails the coverage rule) are dropped. This yields exactly the
    consecutive (t, t+1) supervised pairs counted in Phase 0.

    Args:
        daily: output of `daily_mean_target` (qualifying days only).

    Returns:
        DataFrame indexed by origin day t with column `y_next_day` = PM2.5 mean on t+1.
        Rows are only the valid consecutive pairs.

    Why it matters: requiring t+1 to be the immediate next day prevents the model from
    "predicting" across a multi-day gap, which would silently mix horizons.
    """
    s = daily["pm25_daily_mean"]
    # Look up t+1 by calendar date; NaN if that day isn't a qualifying day.
    next_day_index = daily.index + pd.Timedelta(days=1)
    y_next = s.reindex(next_day_index).values  # aligned to t+1's value, positioned at t

    out = pd.DataFrame(index=daily.index)
    out["y_next_day"] = y_next
    # Drop origin days whose t+1 is not a qualifying consecutive day.
    out = out.dropna(subset=["y_next_day"])
    out.index.name = "origin_day"
    return out
