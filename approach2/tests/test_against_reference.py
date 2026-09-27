"""Validate the rebuilt master against the reference dataset.

Schema-agnostic by design: it discovers the PM2.5 ground-truth column in each
file rather than hard-coding names, because the reference
(data/reference/dhaka_aq_master.csv) and the rebuild
(data/processed/dhaka_aq_master_rebuilt.parquet) may use different naming
lineages (openaq_pm25 vs aq_pm25 vs pm25).

Run:  pytest tests/test_against_reference.py -v
Both files must exist; the test SKIPS (loudly) if either is missing so the
suite can run pre-ingest without false failures.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

REFERENCE = Path("data/reference/dhaka_aq_master.csv")
REBUILT_PARQUET = Path("data/processed/dhaka_aq_master_rebuilt.parquet")
REBUILT_CSV = Path("data/processed/dhaka_aq_master_rebuilt.csv")

# Ground-truth PM2.5 column candidates, most specific first. Excludes model /
# satellite PM2.5 (omaq_pm2_5, acag_pm25) on purpose — those are NOT ground truth.
PM25_PATTERNS = [
    r"^openaq_pm2?_?5$",
    r"^aq_pm2?_?5$",
    r"^pm2?_?5$",
]
EXCLUDE_PATTERNS = [r"omaq", r"acag", r"next", r"lag", r"roll", r"pred"]

# Thresholds — material-drift gates. Tighten if needed; do not weaken.
MAX_MEDIAN_DRIFT_PCT = 15.0      # rebuilt vs reference median PM2.5
MIN_YEAR_COVERAGE_RATIO = 0.70   # rebuilt valid hours per year >= 70% of reference
EXPECTED_CONTINUOUS_YEARS = range(2016, 2026)  # reference is continuous 2016–2026


def _find_pm25_col(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    for pat in PM25_PATTERNS:
        for c in cols:
            if re.match(pat, c, flags=re.IGNORECASE) and not any(
                re.search(x, c, flags=re.IGNORECASE) for x in EXCLUDE_PATTERNS
            ):
                return c
    raise AssertionError(
        f"No ground-truth PM2.5 column found. Columns: {cols}. "
        "If the naming changed, extend PM25_PATTERNS — do not bypass this test."
    )


def _find_time_col(df: pd.DataFrame) -> str:
    for cand in ("datetime_utc", "timestamp_utc", "datetime", "timestamp", "time_utc"):
        if cand in df.columns:
            return cand
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            return c
    raise AssertionError(f"No timestamp column found. Columns: {list(df.columns)}")


def _load(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _yearly_valid_counts(df: pd.DataFrame) -> pd.Series:
    tcol, pcol = _find_time_col(df), _find_pm25_col(df)
    ts = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    valid = df[pcol].notna() & ts.notna()
    return ts[valid].dt.year.value_counts().sort_index()


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    if not REFERENCE.exists():
        pytest.skip(
            f"REFERENCE MISSING: place dhaka_aq_master.csv at {REFERENCE} "
            "(see README — do NOT substitute the gapped xlsx)."
        )
    return _load(REFERENCE)


@pytest.fixture(scope="module")
def rebuilt() -> pd.DataFrame:
    path = REBUILT_PARQUET if REBUILT_PARQUET.exists() else REBUILT_CSV
    if not path.exists():
        pytest.skip("REBUILD MISSING: run `python -m src.ingest.run` first.")
    return _load(path)


def test_reference_is_the_continuous_master_not_the_gapped_file(reference):
    """Guard against validating the pipeline with the wrong file.

    The old xlsx lineage has a 2020–2023 PM2.5 void; the true master is
    continuous 2016–2026. If this fails, the file in data/reference/ is wrong.
    """
    counts = _yearly_valid_counts(reference)
    missing = [y for y in EXPECTED_CONTINUOUS_YEARS if counts.get(y, 0) == 0]
    assert not missing, (
        f"Reference has ZERO valid PM2.5 hours in {missing}. This looks like the "
        "gapped legacy file, not dhaka_aq_master.csv. Replace it before trusting "
        "any validation downstream."
    )


def test_rebuilt_yearly_coverage_matches_reference(reference, rebuilt):
    ref, reb = _yearly_valid_counts(reference), _yearly_valid_counts(rebuilt)
    failures = []
    for year in EXPECTED_CONTINUOUS_YEARS:
        r, b = ref.get(year, 0), reb.get(year, 0)
        if r > 0 and b < MIN_YEAR_COVERAGE_RATIO * r:
            failures.append(f"{year}: rebuilt {b} vs reference {r}")
    assert not failures, "Coverage drift (>30% loss):\n" + "\n".join(failures)


def test_rebuilt_pm25_distribution_matches_reference(reference, rebuilt):
    ref_med = reference[_find_pm25_col(reference)].median()
    reb_med = rebuilt[_find_pm25_col(rebuilt)].median()
    drift = abs(reb_med - ref_med) / ref_med * 100
    assert drift <= MAX_MEDIAN_DRIFT_PCT, (
        f"Median PM2.5 drift {drift:.1f}% (reference {ref_med:.1f}, "
        f"rebuilt {reb_med:.1f}) exceeds {MAX_MEDIAN_DRIFT_PCT}% — investigate "
        "units, station mix, or aggregation before proceeding."
    )


def test_no_forbidden_columns(rebuilt):
    bad = [c for c in rebuilt.columns if "co2" in c.lower() or "ammonia" in c.lower()]
    assert not bad, (
        f"Forbidden columns present: {bad}. There is no CO2 in this project, and "
        "omaq_ammonia is 100% empty by construction."
    )
