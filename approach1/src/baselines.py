"""
baselines.py — the naive forecasts every ML model must beat.

THE BAR (built and reported FIRST)
    1. Persistence: tomorrow = today. pred(t+1) = observed daily mean on day t.
       This is the hardest baseline to beat on an autocorrelated pollutant.
    2. Seasonal climatology: pred(t+1) = the historical mean PM2.5 for that day-of-year,
       computed FROM TRAINING DATA ONLY (never the test fold) — captures the seasonal cycle.
    3. Seasonal-naive (lag-7): pred(t+1) = the value on the same weekday last week.

WHY THIS COMES BEFORE ANY MODEL
    A forecasting paper is judged on SKILL OVER NAIVE PREDICTORS, not raw R². If a fancy
    model cannot beat "tomorrow = today", the model is not earning its complexity.

Each baseline is a pure function of information available through day t (persistence and
lag-7 are literally feature columns; climatology is fit on train only), so none can leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def persistence(df_test: pd.DataFrame) -> np.ndarray:
    """pred(t+1) = pm25 on day t. This is the `pm25_lag_1` feature by construction."""
    return df_test["pm25_lag_1"].to_numpy(float)


def seasonal_naive_lag7(df_test: pd.DataFrame) -> np.ndarray:
    """pred(t+1) = pm25 on the same weekday last week == the `pm25_lag_7` feature."""
    return df_test["pm25_lag_7"].to_numpy(float)


def fit_climatology(df_train: pd.DataFrame) -> dict[int, float]:
    """Learn a day-of-year -> mean-PM2.5 table from TRAIN rows only.

    The target day is origin_day + 1, so we key the climatology on the target day's
    day-of-year. Returns a dict {doy: mean_target}, plus a global fallback under key -1
    for any day-of-year unseen in this train fold.
    """
    target_day = df_train.index + pd.Timedelta(days=1)
    doy = target_day.dayofyear
    y = df_train["y_next_day"].to_numpy(float)
    table = pd.Series(y).groupby(doy).mean().to_dict()
    table[-1] = float(np.mean(y))  # fallback for unseen day-of-year
    return table


def predict_climatology(clim: dict[int, float], df_test: pd.DataFrame) -> np.ndarray:
    """Apply a fitted climatology table to test rows (by target day-of-year)."""
    target_day = df_test.index + pd.Timedelta(days=1)
    doy = target_day.dayofyear
    fallback = clim[-1]
    return np.array([clim.get(int(d), fallback) for d in doy], dtype=float)
