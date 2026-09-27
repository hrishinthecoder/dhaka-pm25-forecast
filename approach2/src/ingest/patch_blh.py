"""Patch the H1-2024 boundary-layer-height (BLH) gap in the processed master.

The gap (2024-01-01 → 2024-06-30, flagged by `era5_blh_imputed`) is a hole in the
Open-Meteo ERA5 *mirror* only — ERA5 BLH exists in the native CDS archive.

PATH A (preferred): pull TRUE ERA5 BLH from the Copernicus CDS and write it into
the gap hours, setting `era5_blh_imputed = 0` for those hours (they are now real
reanalysis). Selected when CDS credentials + the `cdsapi`/`xarray` stack are
available and the fetch succeeds.

PATH B (fallback): recompute the month×hour BLH climatology from TRAIN YEARS ONLY
(≤2023) and refill the gap hours, keeping `era5_blh_imputed = 1`. The previous
build computed this climatology over ALL years (incl. 2025 test) — a temporal
leak. This corrects it.

    python -m src.ingest.patch_blh [--config config/model.yaml]

Idempotent on PATH B (recomputes from the unchanged train-year observations).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.ingest.cds_blh import (SETUP_INSTRUCTIONS, credentials_available,
                                fetch_blh)

BLH_COL = "era5_boundary_layer_height"
FLAG_COL = "era5_blh_imputed"
TIME_COL = "datetime_utc"
PARQUET = Path("data/processed/dhaka_aq_master_rebuilt.parquet")
CSV = Path("data/processed/dhaka_aq_master_rebuilt.csv")

CHECKSUM_WARNING = (
    "WARNING: BLH for the 2024 H1 gap was imputed by TRAIN-ONLY climatology because no CDS "
    "credentials were found. The rebuilt parquet will NOT match models/final/SHA256SUMS, and "
    "its 2024 features will differ from the published model. For EXACT reproduction, set up CDS "
    "credentials (see the instructions above) and re-run the ingest / `python -m src.ingest.patch_blh`."
)


def _train_end(config_path: str) -> str:
    cfg = yaml.safe_load(Path(config_path).read_text())
    return cfg["split"]["train_end"]


def _path_b(df: pd.DataFrame, gap: pd.Series, train_end: str) -> dict:
    """Train-only month×hour climatology refill. Keeps the imputed flag set."""
    ts = pd.to_datetime(df[TIME_COL], utc=True)
    train_obs = (~gap) & (ts.dt.year <= pd.Timestamp(train_end).year)
    base = df.loc[train_obs, BLH_COL]
    mo, hr = ts[train_obs].dt.month, ts[train_obs].dt.hour
    clim = base.groupby([mo.values, hr.values]).mean()
    gmean = float(base.mean())
    g_mo, g_hr = ts[gap].dt.month, ts[gap].dt.hour
    fill = [clim.get((m, h), gmean) for m, h in zip(g_mo, g_hr)]
    df.loc[gap, BLH_COL] = fill
    df.loc[gap, FLAG_COL] = True                       # still imputed, just leak-free now
    return {"path": "B", "source": "train_climatology",
            "train_years": f"<= {pd.Timestamp(train_end).year}",
            "n_patched": int(gap.sum()), "imputed_mean": round(float(pd.Series(fill).mean()), 2)}


def _path_a(df: pd.DataFrame, gap: pd.Series) -> dict:
    """Real ERA5 BLH from CDS. Clears the imputed flag for patched hours."""
    ts = pd.to_datetime(df[TIME_COL], utc=True)
    start = ts[gap].min().strftime("%Y-%m-%d")
    end = (ts[gap].max().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    series = fetch_blh(start, end)                     # hourly UTC, real reanalysis
    # Align by exact UTC hour; require coverage of every gap hour before committing.
    gap_idx = pd.DatetimeIndex(ts[gap])
    vals = series.reindex(gap_idx)
    if vals.isna().any():
        raise RuntimeError(f"CDS returned no value for {int(vals.isna().sum())} gap hours")
    df.loc[gap, BLH_COL] = vals.values
    df.loc[gap, FLAG_COL] = False                      # real data, not imputed
    return {"path": "A", "source": "cds_era5_native",
            "n_patched": int(gap.sum()), "real_mean": round(float(vals.mean()), 2)}


def patch_frame(df: pd.DataFrame, train_end: str = "2023-12-31") -> tuple[pd.DataFrame, dict]:
    """In-place BLH patch on a master frame (datetime_utc as a column).

    Auto-selects PATH A (CDS native, reusing the cached NetCDF if present) when
    credentials exist, else PATH B (train-only climatology) with a loud warning
    that the result will not match the published checksum. Shared by the standalone
    `patch()` and the ingest merge step so both produce identical parquets.
    """
    if FLAG_COL not in df.columns or BLH_COL not in df.columns:
        raise ValueError(f"Master lacks {BLH_COL}/{FLAG_COL}; cannot patch.")
    gap = df[FLAG_COL].fillna(False).astype(bool)
    before_mean = round(float(df.loc[gap, BLH_COL].mean()), 2) if gap.any() else None

    if not gap.any():
        return df, {"path": "none", "source": "no_gap", "n_patched": 0}

    if credentials_available():
        try:
            info = _path_a(df, gap)
        except Exception as exc:                       # noqa: BLE001 — any failure => fallback
            print(f"[patch_blh] PATH A (CDS) failed: {exc}")
            print(f"[patch_blh] {SETUP_INSTRUCTIONS}")
            print(f"[patch_blh] {CHECKSUM_WARNING}")
            info = _path_b(df, gap, train_end)
    else:
        print(f"[patch_blh] {SETUP_INSTRUCTIONS}")
        print(f"[patch_blh] {CHECKSUM_WARNING}")
        info = _path_b(df, gap, train_end)

    info["gap_mean_before"] = before_mean
    info["era5_blh_imputed_total_after"] = int(df[FLAG_COL].fillna(False).sum())
    return df, info


def patch(config_path: str = "config/model.yaml") -> dict:
    if not PARQUET.exists():
        raise FileNotFoundError(f"{PARQUET} missing — run notebook 01 / src.ingest first.")
    df = pd.read_parquet(PARQUET)
    df, info = patch_frame(df, _train_end(config_path))
    df.to_parquet(PARQUET, index=False)
    df.to_csv(CSV, index=False)
    print(f"[patch_blh] {info}")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/model.yaml")
    args = ap.parse_args()
    patch(args.config)


if __name__ == "__main__":
    main()
