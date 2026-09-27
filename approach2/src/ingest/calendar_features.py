"""Calendar-derived flags: is_holiday (Bangladesh) and is_ramadan.

is_holiday uses the `holidays` package (Bangladesh calendar) when installed, and
falls back to a small set of fixed-date national holidays otherwise, so the
notebook still runs top-to-bottom without the optional dependency.

is_ramadan uses an embedded table of Ramadan Gregorian date ranges (Umm al-Qura
approximation) for 2016–2026 — deterministic and dependency-free, which matters
for reproducibility. All dates are evaluated in Asia/Dhaka local time.
"""
from __future__ import annotations

import pandas as pd

# Ramadan (approx Umm al-Qura) — (start_inclusive, end_inclusive), Gregorian.
RAMADAN_RANGES = {
    2016: ("2016-06-06", "2016-07-05"),
    2017: ("2017-05-27", "2017-06-24"),
    2018: ("2018-05-16", "2018-06-14"),
    2019: ("2019-05-06", "2019-06-03"),
    2020: ("2020-04-24", "2020-05-23"),
    2021: ("2021-04-13", "2021-05-12"),
    2022: ("2022-04-02", "2022-05-01"),
    2023: ("2023-03-23", "2023-04-20"),
    2024: ("2024-03-11", "2024-04-09"),
    2025: ("2025-03-01", "2025-03-29"),
    2026: ("2026-02-18", "2026-03-19"),
}

# Fixed-date Bangladesh national holidays (fallback only; month-day).
_FIXED_BD_HOLIDAYS = {
    (2, 21),   # International Mother Language Day
    (3, 17),   # Sheikh Mujib's Birthday
    (3, 26),   # Independence Day
    (4, 14),   # Pohela Boishakh (Bengali New Year)
    (5, 1),    # May Day
    (8, 15),   # National Mourning Day
    (12, 16),  # Victory Day
    (12, 25),  # Christmas
}


def _ramadan_flag(local_dates: pd.DatetimeIndex) -> pd.Series:
    flag = pd.Series(0, index=local_dates, dtype="int8")
    for yr, (s, e) in RAMADAN_RANGES.items():
        mask = (local_dates >= pd.Timestamp(s)) & (local_dates <= pd.Timestamp(e))
        flag.loc[mask] = 1
    return flag


def _holiday_flag(local_dates: pd.DatetimeIndex) -> pd.Series:
    years = range(local_dates.year.min(), local_dates.year.max() + 1)
    try:
        import holidays as _h
        bd = _h.Bangladesh(years=list(years))
        norm = local_dates.normalize()
        return pd.Series([1 if d.date() in bd else 0 for d in norm],
                         index=local_dates, dtype="int8")
    except Exception:
        # dependency missing -> fixed-date fallback
        return pd.Series(
            [(1 if (d.month, d.day) in _FIXED_BD_HOLIDAYS else 0) for d in local_dates],
            index=local_dates, dtype="int8",
        )


def add_holiday_ramadan(idx_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Return a frame indexed by `idx_utc` with is_holiday and is_ramadan (0/1).

    Both flags are computed from the Asia/Dhaka local date of each UTC timestamp.
    """
    # tz-naive local wall-clock dates, so comparisons against naive Ramadan/holiday
    # boundary timestamps are valid (no tz-aware vs tz-naive errors).
    local = pd.DatetimeIndex(idx_utc).tz_convert("Asia/Dhaka").tz_localize(None)
    out = pd.DataFrame(index=idx_utc)
    out["is_holiday"] = _holiday_flag(local).to_numpy()
    out["is_ramadan"] = _ramadan_flag(local).to_numpy()
    return out
