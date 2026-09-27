"""
plot_coverage_split.py — visualize the single-station target's coverage and the
train / walk-forward-CV / final-hold-out split over time.

WHAT THIS DOES
    Rebuilds the daily-mean PM2.5 target from the chosen single reference sensor
    (24434, US diplomatic-post), applying the same local-day + >=18h coverage rule
    the pipeline uses, then draws a timeline: the daily series, the data gaps, and
    the three split regions. It also marks the date the sensor went offline.

WHY IT MATTERS
    After the Phase-0 single-station decision, two things needed a human eye: the
    reduced sample size (N ~ 1,746 pairs) and where the hold-out lands now that the
    sensor stops in 2025-03. This figure makes both visible at a glance and is
    publication-quality for the manuscript's data section.

This script is read-only on the data: it loads the sensor file, computes the daily
target, and writes a PNG to figures/. It changes no project data.

Run:  python reports/plot_coverage_split.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend so it runs on a server without a display
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# --- config (mirrors config/config.yaml; kept literal here so the plot is standalone) ---
BASE = Path(__file__).resolve().parent.parent           # project root (parent of reports/)
SENSOR_FILE = BASE / "raw" / "openaq" / "sensor_24434_pm25.parquet"
OUT_PNG = BASE / "figures" / "coverage_split.png"
LOCAL_TZ = "Asia/Dhaka"   # UTC+6, no DST — "next day" means a citizen's local day
MIN_HOURS = 18            # a local day counts only with >=18 valid hourly observations
HOLDOUT_MONTHS = 12       # final hold-out length...
HOLDOUT_ANCHOR = "last_available"  # ...anchored to the sensor's LAST qualifying day, not wall-clock now


def build_daily_target(sensor_file: Path) -> pd.Series:
    """Load one OpenAQ sensor file and return its local-day mean PM2.5 series.

    Keeps only days with at least MIN_HOURS valid hourly observations — the same
    rule the modeling pipeline uses — so the picture matches the real supervised set.

    Returns a tz-naive daily Series (index = local calendar day) for clean plotting.
    """
    raw = pd.read_parquet(sensor_file, columns=["datetime_utc", "value"]).dropna(subset=["value"])
    # convert UTC -> Dhaka local time, then bucket into local calendar days
    local_day = pd.to_datetime(raw["datetime_utc"], utc=True).dt.tz_convert(LOCAL_TZ)
    frame = pd.DataFrame({"day": local_day.dt.floor("D").dt.tz_localize(None),
                          "value": raw["value"].values})
    grouped = frame.groupby("day")["value"]
    daily_mean = grouped.mean()
    valid_hours = grouped.size()
    qualifying = daily_mean[valid_hours >= MIN_HOURS].sort_index()  # the supervised days
    return qualifying


def split_dates(daily: pd.Series):
    """Compute the hold-out start so the last HOLDOUT_MONTHS of AVAILABLE data are held out.

    Anchoring to the last qualifying day (not 'now') matters because the sensor went
    offline in 2025-03 — a wall-clock-recent window would be empty.
    """
    last_day = daily.index.max()
    holdout_start = last_day - pd.DateOffset(months=HOLDOUT_MONTHS)
    return holdout_start, last_day


def main():
    daily = build_daily_target(SENSOR_FILE)
    holdout_start, last_day = split_dates(daily)

    train_cv = daily[daily.index < holdout_start]
    holdout = daily[daily.index >= holdout_start]
    # consecutive (t, t+1) supervised pairs — the true N for next-day forecasting
    days = pd.Series(daily.index)
    n_pairs = int((days.shift(-1) == days + pd.Timedelta(days=1)).sum())

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 4.6))

    # the daily target itself (thin line shows both signal and gaps)
    ax.plot(daily.index, daily.values, lw=0.6, color="#333333", zorder=3,
            label="Daily-mean PM2.5 (sensor 24434)")

    # split regions
    ax.axvspan(daily.index.min(), holdout_start, color="#2c7fb8", alpha=0.10, zorder=1,
               label=f"Train + walk-forward CV ({len(train_cv)} days)")
    ax.axvspan(holdout_start, last_day, color="#d95f0e", alpha=0.16, zorder=1,
               label=f"Final hold-out, last 12 mo available ({len(holdout)} days)")

    # reference lines: WHO 24-h guideline (15) and a Dhaka-typical 'Unhealthy' marker
    ax.axhline(15, color="#1a9850", ls="--", lw=0.8, alpha=0.7, zorder=2,
               label="WHO 24-h guideline (15 µg/m³)")

    # sensor offline marker
    ax.axvline(last_day, color="#b30000", ls=":", lw=1.2, zorder=4)
    ax.annotate("sensor offline\n" + last_day.strftime("%Y-%m-%d"),
                xy=(last_day, ax.get_ylim()[1] * 0.92),
                xytext=(-95, 0), textcoords="offset points",
                fontsize=8, color="#b30000",
                arrowprops=dict(arrowstyle="->", color="#b30000", lw=0.8))

    ax.set_title("Single-station PM2.5 target — coverage and validation split\n"
                 f"N = {n_pairs} supervised (t, t+1) day pairs  |  "
                 f"span {daily.index.min().date()} → {last_day.date()}",
                 fontsize=11)
    ax.set_xlabel("Date (Asia/Dhaka local day)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved: {OUT_PNG}")
    print(f"qualifying days: {len(daily)} | train+cv: {len(train_cv)} | "
          f"holdout: {len(holdout)} | (t,t+1) pairs: {n_pairs}")
    print(f"holdout window: {holdout_start.date()} -> {last_day.date()}")


if __name__ == "__main__":
    main()
