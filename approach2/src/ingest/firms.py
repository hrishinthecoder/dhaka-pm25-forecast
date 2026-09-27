"""NASA FIRMS active-fire / FRP ingestion (biomass-burning proxy).

FIRMS area CSV API:
  https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{AREA}/{DAYRANGE}/{DATE}
    AREA     = west,south,east,north   (decimal degrees)
    DAYRANGE = 1..5   (the API caps each request at 5 days -> we page)
    DATE     = YYYY-MM-DD start of the window
  Returns CSV; relevant columns: latitude, longitude, acq_date, acq_time, frp.

Product selection: the NRT feed only covers roughly the last ~2 months. For older
windows we fall back to the Standard-Processing archive product (…_SP). Both are
cached under data/raw/firms/.

Output: daily DataFrame indexed by date with firms_count and firms_frp_sum.
"""
from __future__ import annotations

import io
import math
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path("data/raw/firms")
BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
WINDOW_DAYS = 5                # API hard cap (area CSV: "Expects [1..5]")
NRT_LOOKBACK_DAYS = 60         # NRT feed only covers ~last 2 months
MAX_RETRIES = 5
BACKOFF_BASE = 2.0


def _bbox(lat: float, lon: float, radius_km: float) -> str:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 1e-6))
    west, east = lon - dlon, lon + dlon
    south, north = lat - dlat, lat + dlat
    return f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}"


def _archive_product(nrt_source: str) -> str:
    """Map an NRT product to its Standard-Processing archive counterpart."""
    if nrt_source.endswith("_NRT"):
        return nrt_source[:-4] + "_SP"
    return nrt_source


def _pick_source(window_start: pd.Timestamp, nrt_source: str, today: pd.Timestamp) -> str:
    """NRT for recent windows, archive (_SP) for older ones."""
    if window_start >= today - pd.Timedelta(days=NRT_LOOKBACK_DAYS):
        return nrt_source
    return _archive_product(nrt_source)


def _get_csv(map_key: str, source: str, area: str, days: int, date: str) -> str:
    url = f"{BASE}/{map_key}/{source}/{area}/{days}/{date}"
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(BACKOFF_BASE ** attempt)
                continue
            # 400 = bad window for this product (e.g. a date outside the product's
            # availability near the NRT/SP handoff). Don't retry or abort the whole
            # multi-year pull -> treat as "no detections this window" (fire is a
            # minor covariate; missing days become 0 downstream).
            if r.status_code == 400:
                print(f"  FIRMS 400 (no data for window) {source} {date}: {r.text.strip()[:80]}")
                return ""
            r.raise_for_status()
            return r.text
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(BACKOFF_BASE ** attempt)
    raise RuntimeError(f"FIRMS GET failed after {MAX_RETRIES} retries: {url}") from last_exc


def _parse_csv(text: str) -> pd.DataFrame:
    if not text or "latitude" not in text.split("\n", 1)[0].lower():
        return pd.DataFrame(columns=["acq_date", "frp"])
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception:
        return pd.DataFrame(columns=["acq_date", "frp"])
    if "acq_date" not in df.columns:
        return pd.DataFrame(columns=["acq_date", "frp"])
    if "frp" not in df.columns:
        df["frp"] = 0.0
    return df[["acq_date", "frp"]]


def fetch_fire_daily(lat: float, lon: float, radius_km: float, start: str, end: str,
                     map_key: str, nrt_source: str = "VIIRS_SNPP_NRT") -> pd.DataFrame:
    """Daily fire counts + summed FRP within radius_km of (lat, lon).

    Returns a DataFrame indexed by daily UTC timestamp (date at 00:00) with
    columns firms_count and firms_frp_sum. Days with no detections are 0.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    area = _bbox(lat, lon, radius_km)
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)

    start_ts = pd.Timestamp(start).normalize()
    end_excl = pd.Timestamp(end).normalize()
    last_day = end_excl - pd.Timedelta(days=1)

    parts = []
    cur = start_ts
    while cur <= last_day:
        win_end = min(cur + pd.Timedelta(days=WINDOW_DAYS - 1), last_day)
        days = (win_end - cur).days + 1
        source = _pick_source(cur, nrt_source, today)
        date_str = cur.strftime("%Y-%m-%d")
        cache = RAW_DIR / f"{source}_{date_str}_{days}d.csv"
        if cache.exists():
            text = cache.read_text()
        else:
            text = _get_csv(map_key, source, area, days, date_str)
            cache.write_text(text)
        df = _parse_csv(text)
        if len(df):
            parts.append(df)
        cur = win_end + pd.Timedelta(days=1)

    full_idx = pd.date_range(start_ts, last_day, freq="D")
    if not parts:
        return pd.DataFrame({"firms_count": 0.0, "firms_frp_sum": 0.0}, index=full_idx)

    allf = pd.concat(parts, ignore_index=True)
    allf["acq_date"] = pd.to_datetime(allf["acq_date"], errors="coerce")
    allf = allf.dropna(subset=["acq_date"])
    grp = allf.groupby("acq_date").agg(
        firms_count=("frp", "size"),
        firms_frp_sum=("frp", "sum"),
    )
    out = grp.reindex(full_idx).fillna(0.0)
    out.index.name = "date"
    return out
