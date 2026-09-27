"""Open-Meteo ERA5 archive ingestion (REANALYSIS meteorology).

This is ERA5 **reanalysis** at the project centroid — NOT station observations.
Label it as reanalysis everywhere downstream.

Endpoint: https://archive-api.open-meteo.com/v1/archive  (no API key required)
  params: latitude, longitude, start_date, end_date (YYYY-MM-DD inclusive),
          hourly=<comma list>, timezone=UTC
  response: {"hourly": {"time": [...ISO...], "<var>": [...], ...}}

Pulls are chunked by calendar year and cached under data/raw/open_meteo/.
Output columns are prefixed era5_ to match the reference master.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw/open_meteo")
MAX_RETRIES = 5
BACKOFF_BASE = 2.0


def _get(url: str, params: dict) -> dict:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(BACKOFF_BASE ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            time.sleep(BACKOFF_BASE ** attempt)
    raise RuntimeError(f"Open-Meteo GET failed after {MAX_RETRIES} retries: {url}") from last_exc


def _year_bounds(start: str, end: str):
    """Yield (start_date, end_date) inclusive YYYY-MM-DD per calendar year.

    `end` from config is an EXCLUSIVE upper bound, so the last day fetched is
    end - 1 day.
    """
    start_ts = pd.Timestamp(start)
    end_excl = pd.Timestamp(end)
    last_day = (end_excl - pd.Timedelta(days=1)).normalize()
    cur = start_ts.normalize()
    while cur <= last_day:
        yr_end = min(pd.Timestamp(f"{cur.year}-12-31"), last_day)
        yield cur.strftime("%Y-%m-%d"), yr_end.strftime("%Y-%m-%d")
        cur = (yr_end + pd.Timedelta(days=1)).normalize()


def fetch_era5_hourly(lat: float, lon: float, start: str, end: str,
                      hourly_vars: list[str],
                      endpoint: str = "https://archive-api.open-meteo.com/v1/archive"
                      ) -> pd.DataFrame:
    """Return UTC-indexed hourly ERA5 reanalysis frame (columns era5_<var>)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for d_from, d_to in _year_bounds(start, end):
        cache = RAW_DIR / f"era5_{d_from}_{d_to}.json"
        if cache.exists():
            js = json.loads(cache.read_text())
        else:
            js = _get(endpoint, {
                "latitude": lat, "longitude": lon,
                "start_date": d_from, "end_date": d_to,
                "hourly": ",".join(hourly_vars),
                "timezone": "UTC",
            })
            cache.write_text(json.dumps(js))
        hourly = js.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            continue
        df = pd.DataFrame({v: hourly.get(v) for v in hourly_vars})
        df.index = pd.to_datetime(times, utc=True)
        frames.append(df)
    if not frames:
        raise RuntimeError("Open-Meteo ERA5 returned no data for the requested range.")
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out.columns = [f"era5_{c}" for c in out.columns]
    out.index.name = "datetime_utc"
    return out
