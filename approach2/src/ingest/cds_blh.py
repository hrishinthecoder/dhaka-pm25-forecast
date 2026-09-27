"""PATH A — pull TRUE ERA5 boundary-layer height from the Copernicus CDS.

The 2024-01-01 → 2024-06-30 BLH hole exists ONLY in the Open-Meteo *mirror* of
ERA5. The variable `boundary_layer_height` is fully available in the native CDS
archive (`reanalysis-era5-single-levels`). This module pulls it directly via the
`cdsapi` package — the new **CDS-Beta** endpoint (https://cds.climate.copernicus.eu/api,
request syntax changed in 2024: `product_type` is a list, `data_format`/
`download_format` replace the old `format` key) — and returns an hourly UTC series
at the Dhaka gridpoint so the gap can be patched with REAL reanalysis values.

Requires: `cdsapi`, `xarray`, `netCDF4`, and CDS credentials (`~/.cdsapirc`,
a project-local `.cdsapirc`, or the `CDSAPI_KEY`/`CDSAPI_URL` env vars). Register
and accept the ERA5 licence at cds.climate.copernicus.eu before first use.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DATASET = "reanalysis-era5-single-levels"
VARIABLE = "boundary_layer_height"
LAT, LON = 23.8103, 90.4125                 # Dhaka (US embassy reference point)
# Small box around the point on the ERA5 native 0.25° grid: [North, West, South, East].
AREA = [24.0, 90.25, 23.5, 90.75]

SETUP_INSTRUCTIONS = (
    "CDS credentials not found — PATH A (native ERA5 pull) skipped.\n"
    "To enable it:\n"
    "  1. Register at https://cds.climate.copernicus.eu and log in.\n"
    "  2. Accept the 'ERA5 hourly data on single levels' licence on the dataset page.\n"
    "  3. Create ~/.cdsapirc with:\n"
    "         url: https://cds.climate.copernicus.eu/api\n"
    "         key: <your-personal-access-token>\n"
    "     (or export CDSAPI_URL / CDSAPI_KEY).\n"
    "Falling back to PATH B (train-only month×hour climatology)."
)


def credentials_available() -> bool:
    """True if a CDS token is reachable via env vars or an rc file."""
    if os.environ.get("CDSAPI_KEY"):
        return True
    return any(p.exists() for p in (Path.home() / ".cdsapirc", Path(".cdsapirc")))


def fetch_blh(start: str, end: str,
              target: str = "data/interim/era5_blh_cds.nc") -> pd.Series:
    """Retrieve hourly BLH for [start, end) at the Dhaka gridpoint.

    Returns a tz-aware (UTC) hourly `pd.Series` named `era5_boundary_layer_height`.
    Raises on any cdsapi / network / parse failure so the caller can fall back.
    """
    import xarray as xr        # noqa: PLC0415

    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    months = sorted({d.strftime("%m") for d in pd.date_range(s, e, freq="MS").union([s])})
    # Be permissive on months: cover the whole [start, end) span.
    span = pd.period_range(s.tz_localize(None), e.tz_localize(None), freq="M")
    years = sorted({p.strftime("%Y") for p in span})
    months = sorted({p.strftime("%m") for p in span})

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    if Path(target).exists():
        # Reuse the cached NetCDF (the gap window is fixed) instead of re-requesting.
        print(f"[cds_blh] reusing cached NetCDF {target} (no CDS request)")
    else:
        import cdsapi          # noqa: PLC0415 (import here so missing pkg => fallback)
        client = cdsapi.Client()
        client.retrieve(
            DATASET,
            {
                "product_type": ["reanalysis"],
                "variable": [VARIABLE],
                "year": years,
                "month": months,
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": AREA,
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            target,
        )

    ds = xr.open_dataset(target)
    var = VARIABLE if VARIABLE in ds.variables else ("blh" if "blh" in ds.variables else None)
    if var is None:                                  # pick the lone data variable
        data_vars = [v for v in ds.data_vars]
        if len(data_vars) != 1:
            raise ValueError(f"Cannot identify BLH variable in CDS file: {data_vars}")
        var = data_vars[0]
    point = ds[var].sel(latitude=LAT, longitude=LON, method="nearest")

    ser = point.to_series()
    # New CDS files index on 'valid_time'; older on 'time'. to_series() gives a
    # DatetimeIndex either way once we drop the (now scalar) lat/lon levels.
    if isinstance(ser.index, pd.MultiIndex):
        time_level = [n for n in ser.index.names
                      if n and ("time" in n.lower())]
        ser = ser.reset_index()
        tcol = time_level[0] if time_level else ser.columns[0]
        ser = pd.Series(ser[var].values,
                        index=pd.DatetimeIndex(ser[tcol]), name=var)
    ser.index = pd.DatetimeIndex(ser.index)
    if ser.index.tz is None:
        ser.index = ser.index.tz_localize("UTC")
    else:
        ser.index = ser.index.tz_convert("UTC")
    ser = ser.sort_index().rename("era5_boundary_layer_height")
    ds.close()
    return ser.loc[(ser.index >= s) & (ser.index < e)]
