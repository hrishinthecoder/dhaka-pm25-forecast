"""
evaluate.py — forecasting metrics and skill scores.

METRICS
    RMSE, MAE, sMAPE, R² — standard regression error measures.
    SKILL SCORE vs a baseline = 1 - (model_error / baseline_error). It answers the only
    question a forecasting paper really cares about: "is the model better than doing nothing
    clever?" Skill > 0 means the model beats the baseline; skill <= 0 means it does not.

WHY SKILL, NOT JUST R²
    A high R² can still lose to naive persistence on a strongly autocorrelated series.
    Reporting skill vs persistence and vs climatology forces an honest comparison.
"""

from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root mean squared error (same units as PM2.5, penalises large misses)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error (robust, same units as PM2.5)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true, y_pred) -> float:
    """Symmetric mean absolute percentage error (%), bounded and scale-free.

    Uses the 2*|a-b|/(|a|+|b|) form; guards against divide-by-zero on near-zero days.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-9
    return float(np.mean(2.0 * np.abs(y_pred - y_true)[mask] / denom[mask]) * 100.0)


def r2(y_true, y_pred) -> float:
    """Coefficient of determination (fraction of variance explained)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def all_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute the full metric set for one (y_true, y_pred) pair."""
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred),
            "smape": smape(y_true, y_pred), "r2": r2(y_true, y_pred)}


def skill_score(model_error: float, baseline_error: float) -> float:
    """Skill = 1 - model_error/baseline_error. Positive => model beats the baseline."""
    if baseline_error <= 0:
        return float("nan")
    return float(1.0 - model_error / baseline_error)
