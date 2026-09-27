"""Assemble the modeling master from the ingested sources.

Timeline = a continuous hourly UTC index over the config date range (matches the
reference master, which is continuous even where PM2.5 is missing). Everything is
left-joined onto it:

    openaq_*  (ground truth + covariates, collapsed across stations by hourly mean)
    era5_*    (ERA5 reanalysis meteorology)
    firms_*   (daily fire counts / FRP, broadcast to every hour of the day)
    calendar  (hour/dow/month/weekend/rush + cyclical encodings)
    is_holiday, is_ramadan

Output: data/processed/dhaka_aq_master_rebuilt.parquet (+ .csv) with datetime_utc
and datetime_local columns. PM2.5 ground truth lives in `openaq_pm25`.

TARGET (v2.0, single-station): the modeling target `openaq_pm25` is ONE reference
monitor — OpenAQ sensor 24434 (US Diplomatic Post, ~2 km from centre) — NOT a mean
across stations. `_collapse_stations` below averages covariates across stations for
provenance, but covariates are excluded as predictors, so that pooling never
reaches the model. The published single-station master is built by
`fusion/swap_target.py` (sensor 24434 swapped into `openaq_pm25`). If a full
re-ingest is ever run, the target column MUST come from sensor 24434 alone (filter
`station_id == TARGET_SENSOR_ID`), never the pooled mean.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")
OUT_PARQUET = PROCESSED / "dhaka_aq_master_rebuilt.parquet"
OUT_CSV = PROCESSED / "dhaka_aq_master_rebuilt.csv"

PROTECTED = {"openaq_pm25"}  # never auto-dropped, even if temporarily empty
TARGET_SENSOR_ID = 24434     # single-station target monitor (US Diplomatic Post)


def _collapse_stations(aq_hourly: pd.DataFrame) -> pd.DataFrame:
    """Mean across stations per hour; drop the station_id bookkeeping column.

    NOTE (v2.0): this cross-station mean applies to COVARIATES only. The PM2.5
    target is single-sensor (TARGET_SENSOR_ID); on a full re-ingest, build
    `openaq_pm25` from that sensor alone, not from this pooled mean. See module
    docstring and fusion/swap_target.py.
    """
    df = aq_hourly.drop(columns=[c for c in ("station_id",) if c in aq_hourly.columns])
    return df.groupby(level=0).mean()


BLH_COL = "era5_boundary_layer_height"


def _impute_blh(master: pd.DataFrame, train_end: str = "2023-12-31") -> pd.DataFrame:
    """Fill ERA5 boundary-layer-height gaps with month×hour climatology.

    The 2024-01-01..2024-06-30 hole is a gap in the **Open-Meteo ERA5 mirror**, not
    in the source: the variable exists in the native CDS archive (see
    `src.ingest.cds_blh` for the direct CDS pull). When CDS is unavailable we
    impute the gap with the mean BLH per (calendar-month, hour-of-day) — BLH has a
    strong diurnal+seasonal cycle, so this is defensible.

    Leakage control: the climatology is computed from **training years only**
    (index ≤ `train_end`, default 2023-12-31). Using later years (2024 val / 2025
    test) would let future information flow backward into the imputed validation
    window. A boolean `era5_blh_imputed` column flags every filled row.
    No-op (flag all False) when BLH is fully present.
    """
    if BLH_COL not in master.columns:
        return master
    miss = master[BLH_COL].isna()
    master["era5_blh_imputed"] = miss.values
    if not miss.any() or master[BLH_COL].notna().sum() == 0:
        return master
    train_mask = master.index <= pd.Timestamp(train_end, tz=master.index.tz)
    train_blh = master[BLH_COL][train_mask]
    clim = train_blh.groupby([train_blh.index.month, train_blh.index.hour]).mean()
    global_mean = float(train_blh.mean())
    fill = [clim.get((master.index[i].month, master.index[i].hour), global_mean)
            for i in miss.values.nonzero()[0]]
    master.loc[miss, BLH_COL] = fill
    return master


def _broadcast_daily(fire_daily: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex a daily frame onto an hourly index (same values for all 24 h)."""
    fd = fire_daily.copy()
    if fd.index.tz is None:
        fd.index = pd.to_datetime(fd.index).tz_localize("UTC")
    else:
        fd.index = fd.index.tz_convert("UTC")
    day_keys = idx.normalize()
    out = fd.reindex(day_keys)
    out.index = idx
    return out


def build_master(aq_hourly: pd.DataFrame,
                 met_hourly: pd.DataFrame,
                 fire_daily: pd.DataFrame,
                 add_calendar,
                 add_holiday_ramadan,
                 start: str, end: str,
                 drop_columns: list[str] | None = None) -> pd.DataFrame:
    """Build, save, and return the rebuilt master frame."""
    drop_columns = drop_columns or []
    idx = pd.date_range(start=start, end=end, freq="h", tz="UTC", inclusive="left")
    idx.name = "datetime_utc"

    aq = _collapse_stations(aq_hourly).reindex(idx)
    met = met_hourly.reindex(idx)
    fire = _broadcast_daily(fire_daily, idx)
    cal = add_calendar(idx)
    hol = add_holiday_ramadan(idx)

    master = pd.concat([aq, met, fire, cal, hol], axis=1)

    # Fill the ERA5 boundary-layer-height H1-2024 gap (climatology + flag col).
    master = _impute_blh(master)

    # Drop configured + fully-empty columns (the "omaq_ammonia-like" rule),
    # but never drop the protected ground-truth target.
    to_drop = {c for c in drop_columns if c in master.columns}
    to_drop |= {c for c in master.columns
                if c not in PROTECTED and master[c].isna().all()}
    master = master.drop(columns=sorted(to_drop))

    # Hard guard: no CO2, no ammonia — ever.
    forbidden = [c for c in master.columns
                 if "co2" in c.lower() or "ammonia" in c.lower()]
    if forbidden:
        raise ValueError(f"Forbidden columns produced: {forbidden}")

    master = master.reset_index()  # datetime_utc becomes a column
    master.insert(1, "datetime_local",
                  master["datetime_utc"].dt.tz_convert("Asia/Dhaka"))

    # Upgrade the H1-2024 BLH gap to native ERA5 from the CDS when credentials exist
    # (reuses the cached NetCDF). Without credentials this keeps the train-only
    # climatology from _impute_blh and warns that the parquet won't match SHA256SUMS.
    from src.ingest.patch_blh import patch_frame      # local import: keeps cdsapi lazy
    master, _blh_info = patch_frame(master)
    print(f"[merge] BLH patch: {_blh_info}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    master.to_parquet(OUT_PARQUET, index=False)
    master.to_csv(OUT_CSV, index=False)
    return master
