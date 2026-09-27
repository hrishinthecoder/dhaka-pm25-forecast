"""OpenAQ v3 hourly ground-truth ingestion for Dhaka PM2.5.

Target = ground-level **PM2.5** (µg/m³). pm10/o3/co/no2/so2 are covariates.
There is NO CO2 anywhere in this project.

Endpoints (confirmed against current OpenAQ v3 docs, 2026-06):
  * GET /v3/locations/{id}                -> results[0].sensors[] {id, parameter.name}
  * GET /v3/sensors/{id}/hours            -> hourly mean per sensor
        params: datetime_from, datetime_to (ISO8601), limit (<=1000), page
                NOTE: the v3 hours endpoint uses datetime_from/datetime_to, NOT
                date_from/date_to. The latter are silently ignored -> every window
                returns the sensor's default history (the 2016-only bug).
        results[]: {value, parameter:{name,units}, period:{datetimeFrom:{utc}, ...}}
        meta.found: total rows for the query (paginate until exhausted)
Auth: header  X-API-Key: <OPENAQ_API_KEY>   (NOT a query param).

Raw JSON pages are cached under data/raw/openaq/ so a "Restart & run all" after
the first pull is fast and offline-resilient.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.openaq.org/v3"
RAW_DIR = Path("data/raw/openaq")
PAGE_LIMIT = 1000              # OpenAQ v3 hard cap per page
MAX_RETRIES = 5
BACKOFF_BASE = 2.0            # seconds; exponential
ABANDON_AFTER = 3            # consecutive failed windows -> give up on that sensor
TIMEOUT_RETRIES = 2          # 408 attempts per window before fast-failing it
TIMEOUT_WAIT = 1.0           # seconds between 408 attempts (short; retry rarely helps)

# Map OpenAQ parameter names -> our master column names (openaq_ prefix).
# Only these are requested; anything else a station reports is ignored.
PARAM_COLS = {
    "pm25": "openaq_pm25",
    "pm10": "openaq_pm10",
    "o3": "openaq_o3",
    "co": "openaq_co",       # combustion tracer covariate (NOT CO2)
    "no2": "openaq_no2",
    "so2": "openaq_so2",
}


def _get(url: str, api_key: str, params: dict | None = None) -> dict:
    """GET with X-API-Key.

    429/5xx/network errors get the full MAX_RETRIES with exponential backoff.
    408 (server timed out aggregating) is treated as a FAST-FAIL: re-issuing the
    identical heavy query just times out again, so we try it only TIMEOUT_RETRIES
    times with short waits, then give up on the window. The caller abandons the
    sensor after a few consecutive give-ups (see ABANDON_AFTER).
    """
    headers = {"X-API-Key": api_key}
    last_exc: Exception | None = None
    attempt = 0
    timeout_hits = 0
    while attempt < MAX_RETRIES:
        try:
            r = requests.get(url, headers=headers, params=params, timeout=60)
            if r.status_code == 408:
                timeout_hits += 1
                if timeout_hits >= TIMEOUT_RETRIES:
                    raise RuntimeError(
                        f"OpenAQ 408 timeout x{timeout_hits} (query too heavy / "
                        f"no data for range): {url}"
                    )
                time.sleep(TIMEOUT_WAIT)
                continue                      # does NOT consume the main retry budget
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("Retry-After", BACKOFF_BASE ** attempt))
                time.sleep(wait)
                attempt += 1
                continue
            r.raise_for_status()
            return r.json()
        except RuntimeError:
            raise                             # 408 fast-fail -> bubble straight up
        except (requests.RequestException, ValueError) as exc:  # noqa: PERF203
            last_exc = exc
            time.sleep(BACKOFF_BASE ** attempt)
            attempt += 1
    raise RuntimeError(f"OpenAQ GET failed after {MAX_RETRIES} retries: {url}") from last_exc


def list_sensors(location_id: int, api_key: str, wanted: set[str]) -> list[dict]:
    """Return [{sensor_id, parameter}] for the wanted parameters at a location."""
    js = _get(f"{BASE}/locations/{location_id}", api_key)
    results = js.get("results", [])
    if not results:
        return []
    sensors = results[0].get("sensors", []) or []
    out = []
    for s in sensors:
        pname = (s.get("parameter") or {}).get("name")
        if pname in wanted:
            out.append({"sensor_id": s.get("id"), "parameter": pname})
    return out


def _month_windows(start: str, end: str):
    """Yield (date_from, date_to) ISO strings chunked by calendar MONTH.

    Monthly windows keep each query to <=744 hourly rows (one page) and a small
    time range, which avoids OpenAQ 408 Request Timeout on heavy aggregations.
    `end` is treated as an EXCLUSIVE upper bound (matches sources.yaml).
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    cur = start_ts
    while cur < end_ts:
        nxt = min(cur + pd.offsets.MonthBegin(1), end_ts)
        if nxt <= cur:  # guard if start is mid-month
            nxt = min((cur + pd.offsets.MonthBegin(1)).normalize(), end_ts)
        yield cur.isoformat(), nxt.isoformat()
        cur = nxt


def _fetch_sensor_window(sensor_id: int, date_from: str, date_to: str,
                         api_key: str) -> list[dict]:
    """All hourly rows for one sensor in one window, paginated, cached to disk."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"sensor{sensor_id}_{date_from[:10]}_{date_to[:10]}"
    cache = RAW_DIR / f"{tag}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    rows: list[dict] = []
    page = 1
    while True:
        js = _get(
            f"{BASE}/sensors/{sensor_id}/hours",
            api_key,
            params={"datetime_from": date_from, "datetime_to": date_to,
                    "limit": PAGE_LIMIT, "page": page},
        )
        results = js.get("results", []) or []
        rows.extend(results)
        if len(results) < PAGE_LIMIT:
            break
        page += 1
        if page > 1000:  # safety valve against runaway paging
            break
    cache.write_text(json.dumps(rows))
    return rows


def _parse_rows(rows: list[dict], col: str) -> pd.Series:
    """Turn raw OpenAQ hourly rows into a UTC-indexed Series named `col`."""
    recs = []
    for r in rows:
        try:
            ts = r["period"]["datetimeFrom"]["utc"]
            val = r.get("value")
        except (KeyError, TypeError):
            continue
        if val is not None:
            recs.append((ts, val))
    if not recs:
        return pd.Series(name=col, dtype="float64")
    s = pd.DataFrame(recs, columns=["ts", col])
    idx = pd.to_datetime(s["ts"], utc=True)
    s = s.set_index(idx)[col]
    # collapse any duplicate hour rows (e.g. window overlaps) by mean
    return s.groupby(level=0).mean()


def fetch_openaq_hourly(stations: pd.DataFrame, start: str, end: str,
                        api_key: str, parameters: list[str]) -> pd.DataFrame:
    """Tidy hourly OpenAQ frame for every station in `stations`.

    Parameters
    ----------
    stations : DataFrame with at least an 'id' column (location ids), as produced
        by station auto-discovery / config/stations_resolved.yaml.
    parameters : list of OpenAQ parameter names from config (pm25, pm10, ...).

    Returns
    -------
    Long/tidy DataFrame indexed by UTC timestamp with columns:
        station_id, openaq_pm25, openaq_pm10, openaq_o3, openaq_co, ...
    One row per (timestamp, station). The merge step collapses across stations for
    COVARIATES only; the PM2.5 target is single-sensor 24434 (see src/ingest/merge.py).
    """
    wanted = {p for p in parameters if p in PARAM_COLS}
    frames = []
    for _, st in stations.iterrows():
        loc_id = int(st["id"])
        sensors = list_sensors(loc_id, api_key, wanted)
        if not sensors:
            continue
        per_param = {}
        for sen in sensors:
            col = PARAM_COLS[sen["parameter"]]
            chunks = []
            consec_fail = 0
            for d_from, d_to in _month_windows(start, end):
                try:
                    raw = _fetch_sensor_window(sen["sensor_id"], d_from, d_to, api_key)
                except RuntimeError as exc:
                    # one window stuck (e.g. persistent 408) -> warn and skip it.
                    consec_fail += 1
                    print(f"  skip sensor {sen['sensor_id']} {d_from[:7]}: {exc}")
                    # A sensor that fails ABANDON_AFTER windows in a row is dead for
                    # this range (e.g. a recent sensor queried for old years that the
                    # server can't aggregate -> repeated 408). Abandon it so we don't
                    # burn ~31s/window of retries over all 120 windows.
                    if consec_fail >= ABANDON_AFTER:
                        print(f"  ABANDON sensor {sen['sensor_id']}: "
                              f"{consec_fail} consecutive failures, skipping rest.")
                        break
                    continue
                consec_fail = 0
                chunks.append(_parse_rows(raw, col))
            if chunks:
                per_param[col] = pd.concat(chunks).groupby(level=0).mean()
        if not per_param:
            continue
        wide = pd.concat(per_param.values(), axis=1)
        wide.columns = list(per_param.keys())
        wide["station_id"] = loc_id
        frames.append(wide)
    if not frames:
        raise RuntimeError(
            "OpenAQ returned no measurements for any resolved station. "
            "Check OPENAQ_API_KEY, the date range, and stations_resolved.yaml."
        )
    out = pd.concat(frames)
    out.index.name = "datetime_utc"
    return out.sort_index()
