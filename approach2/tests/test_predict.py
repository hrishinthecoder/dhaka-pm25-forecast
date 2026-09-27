"""Unit tests for the frozen inference entrypoint (src.predict).

Known day-t dates -> expected next-day PM2.5 + AQI from the frozen XGBoost model.
SKIPS (loudly) if the freeze pack is absent so the suite runs pre-freeze.
Reference values produced by `python -m src.freeze_model` on the single-station
(sensor 24434) CDS-patched master (seed 42, 45 features); regenerate them if the
frozen model is re-cut. Dates must fall in the single-station range (ends 2025-03-24).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.predict import FeatureUnavailable, predict_for_date

FINAL = Path("models/final")

pytestmark = pytest.mark.skipif(
    not (FINAL / "xgboost_final.joblib").exists(),
    reason="frozen model missing — run `python -m src.freeze_model` first.")


def test_known_date_monsoon():
    out = predict_for_date("2024-07-15")
    assert out["target_date"] == "2024-07-16"
    assert out["pm25_next_day_ugm3"] == pytest.approx(27.5, abs=0.1)
    assert out["aqi_category"] == "Moderate"


def test_known_date_winter_episode():
    out = predict_for_date("2025-01-15")
    assert out["target_date"] == "2025-01-16"
    assert out["pm25_next_day_ugm3"] == pytest.approx(180.2, abs=0.1)
    assert out["aqi_category"] == "Very Unhealthy"


def test_refuses_out_of_range_date():
    with pytest.raises(FeatureUnavailable, match="outside the available"):
        predict_for_date("2099-01-01")
