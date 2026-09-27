"""
data_loader.py — load the two raw inputs the pipeline reads.

INPUTS
    1. The merged master (`processed/dhaka_aq_master.parquet`): hourly ERA5 meteorology,
       CAMS chemistry, FIRMS fire, ACAG anchor, etc. We use it ONLY for FEATURES.
    2. The single chosen target sensor (`raw/openaq/sensor_24434_pm25.parquet`): the
       US diplomatic-post reference monitor. This — not the pooled `openaq_pm25` column —
       is the prediction TARGET (Phase-0 decision, 2026-06-10).

WHY THE SPLIT MATTERS
    Phase 0 proved the master's `openaq_pm25` is a spatial mean over ~14 sites, which
    breaks the single-station framing. Pulling the target straight from one sensor file
    keeps the target honest and single-point, while still using the rich master columns
    as predictors.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import PROJECT_ROOT


def load_master(cfg: dict) -> pd.DataFrame:
    """Load the hourly merged master used for FEATURES only.

    Returns a DataFrame with a tz-aware `datetime_utc` and `datetime_local` plus all
    ERA5/CAMS/FIRMS/ACAG columns. The pooled `openaq_pm25` column is retained only as a
    city-mean reference — it is NOT the target.
    """
    path = PROJECT_ROOT / cfg["paths"]["master_parquet"]
    df = pd.read_parquet(path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df


def load_target_sensor(cfg: dict) -> pd.Series:
    """Load the single-station target as an hourly PM2.5 Series (UTC-indexed).

    Steps (and why):
        - Read the raw OpenAQ sensor file for sensor 24434.
        - Floor each measurement to its UTC hour and average any sub-hourly duplicates,
          so we get at most one value per hour (matches the master's hourly grid).
        - Apply the config physical bounds [pm25_min, pm25_max] as a hard sanity filter;
          out-of-range readings become NaN (Phase 0 found none, but we enforce it anyway).

    Returns:
        pd.Series of hourly PM2.5 (µg/m³) indexed by tz-aware UTC hour, named 'pm25'.
        NaN hours are dropped (the sensor simply did not report then).
    """
    sensor_path = PROJECT_ROOT / cfg["target"]["sensor_file"]
    raw = pd.read_parquet(sensor_path, columns=["datetime_utc", "value"])
    hour = pd.to_datetime(raw["datetime_utc"], utc=True).dt.floor("h")
    # Average any duplicate sub-hourly samples within the same hour.
    series = pd.Series(raw["value"].values, index=hour).groupby(level=0).mean()
    # Hard physical-bounds filter from config (defensive; Phase 0 saw 0 violations).
    lo = cfg["data"]["physical_bounds"]["pm25_min"]
    hi = cfg["data"]["physical_bounds"]["pm25_max"]
    series = series[(series >= lo) & (series <= hi)]
    series.name = "pm25"
    series.index.name = "datetime_utc"
    return series.sort_index()
