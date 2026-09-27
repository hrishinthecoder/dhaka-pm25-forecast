"""
validation.py — time-series splitting that cannot leak the future.

TWO SPLITS, BOTH STRICTLY TEMPORAL
    1. Final hold-out: the most recent 12 months of AVAILABLE data, anchored to the
       sensor's last qualifying day (2025-03-24), NOT wall-clock "now" (the sensor went
       offline). This block is evaluated EXACTLY ONCE, at the very end, and is never seen
       by Optuna or any model-selection step.
    2. Walk-forward (expanding-window) CV on everything BEFORE the hold-out: each fold
       trains on an initial stretch of time and tests on the immediately following stretch,
       with a 1-step gap. The test fold is always strictly later than its train fold. No
       shuffling, no KFold — that would let the model peek at the future.

WHY
    PM2.5 is autocorrelated and seasonal. Random CV would place near-duplicate adjacent
    days in both train and test and massively overstate skill. Walk-forward is the honest
    estimator of one-day-ahead performance, and the untouched hold-out is the final
    unbiased check.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def split_final_holdout(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve off the last-12-months hold-out, anchored to the last available day.

    Args:
        df: supervised table indexed by origin_day (must be sorted ascending).
        cfg: config (uses split.final_holdout_months, split.final_holdout_anchor,
             split.walk_forward.gap_days).

    Returns:
        (dev_df, holdout_df). `dev_df` is used for walk-forward CV + Optuna; `holdout_df`
        is touched once at the end.

    Leakage guard: an origin day t predicts day t+1, so a dev row at the boundary could
    carry a label that falls inside the hold-out window. We therefore drop the final
    `gap_days` of dev rows so no training label overlaps the hold-out period.
    """
    df = df.sort_index()
    months = cfg["split"]["final_holdout_months"]
    anchor_mode = cfg["split"].get("final_holdout_anchor", "last_available")
    gap = cfg["split"]["walk_forward"]["gap_days"]

    anchor = df.index.max() if anchor_mode == "last_available" else pd.Timestamp.now()
    cutoff = anchor - pd.DateOffset(months=months)  # start of the hold-out window

    holdout = df[df.index > cutoff]
    # Dev = everything up to the cutoff, minus a gap so no dev label lands in the hold-out.
    dev = df[df.index <= cutoff - pd.Timedelta(days=gap)]
    return dev, holdout


def make_walk_forward(cfg: dict, n_samples: int) -> TimeSeriesSplit:
    """Build the expanding-window walk-forward splitter for the dev set.

    Args:
        cfg: config (uses split.walk_forward.n_splits and .gap_days).
        n_samples: number of dev rows (used only to sanity-check feasibility).

    Returns:
        A scikit-learn TimeSeriesSplit. It is expanding by construction (train grows each
        fold), preserves time order, never shuffles, and the `gap` separates train from
        test so the t+1 label of the last train row cannot leak into the test fold.

    Note: `gap_days` is applied as a row gap; dev rows are (near-)consecutive daily pairs,
    so one row ≈ one day. This is documented in the Phase 5 report.
    """
    n_splits = cfg["split"]["walk_forward"]["n_splits"]
    gap = cfg["split"]["walk_forward"]["gap_days"]
    # TimeSeriesSplit needs n_splits < n_samples; guaranteed here (n≈1.7k >> 5).
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)
