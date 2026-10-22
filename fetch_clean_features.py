r"""
fetch_clean_features.py  -  RADIANT-Net revision, clean feature extraction
==========================================================================

Rebuilds the event features for the SAME 500 catalogue events in
data/radiant_real_disaster_data.csv, using only real, documented data:

  * Open-Meteo Historical Weather API (ERA5 / ERA5-Land reanalysis)
      daily:  precipitation_sum, temperature_2m_mean, relative_humidity_2m_mean,
              wind_speed_10m_max, surface_pressure_mean, et0_fao_evapotranspiration
      hourly: soil_moisture_0_to_7cm
  * Open-Meteo Elevation API (Copernicus GLO-90 DEM)
      elevation at the event point and at four neighbours -> terrain slope

Differences from fetch_real_data.py (deliberate, for the reviewers):
  * NO random noise is added to any value.
  * NO fallback constants (20 degC, 1013 hPa, 40 % ...): missing stays missing.
  * NO feature is generated per hazard class, and nothing is copied from the
    catalogue record (no magnitude, storm severity, alert score, damage).
  * Slope comes from a real DEM, not from a latitude rule.

Every number written to the output can be re-derived from the two public APIs.

Run (from the RADIANT-Net folder, in the Cursor terminal):
    .\.venv\Scripts\python.exe fetch_clean_features.py
It takes about 10-20 minutes. It caches every API response in
data/cache/clean_openmeteo/, so if it stops you can simply run it again and it
continues where it left off.

Output:
    data/radiant_clean_openmeteo_data.csv
    data/radiant_clean_openmeteo_data_log.txt   (settings, counts, failures)
Send both files back for the analysis.
"""

from __future__ import annotations

import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" if (HERE / "data" / "radiant_real_disaster_data.csv").exists() else HERE  # project folder or flat repository
INPUT_CSV = DATA_DIR / "radiant_real_disaster_data.csv"
OUTPUT_CSV = DATA_DIR / "radiant_clean_openmeteo_data.csv"
LOG_FILE = DATA_DIR / "radiant_clean_openmeteo_data_log.txt"
CACHE = DATA_DIR / "cache" / "clean_openmeteo"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
USER_AGENT = "RADIANT-Net-revision/1.0 (academic research; SR University)"

# Feature window around the event date (same as the original script):
WINDOW_BEFORE = 2   # days before the event date
WINDOW_AFTER = 1    # days after the event date
# Antecedent window for accumulated rainfall (floods / landslides depend on it):
ANTECEDENT_DAYS = 30

DAILY_VARS = [
    "precipitation_sum",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "surface_pressure_mean",
    "et0_fao_evapotranspiration",
]
HOURLY_VARS = ["soil_moisture_0_to_7cm"]

SLOPE_OFFSET_DEG = 0.0025  # ~280 m; neighbours N, S, E, W for central differences
EARTH_M_PER_DEG = 111_320.0


# --------------------------------------------------------------------------- #
def http_get_json(url: str, retries: int = 8) -> dict:
    """GET with polite retries (Open-Meteo free tier limits calls per minute and per hour)."""
    delay = 5.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code == 429 or err.code >= 500:
                print(f"    HTTP {err.code}, retrying in {delay:.0f}s ...")
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as err:
            print(f"    network error ({err}), retrying in {delay:.0f}s ...")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


def cached(name: str, url: str) -> dict:
    path = CACHE / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = http_get_json(url)
    path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(0.15)
    return data


def mean_valid(values) -> float:
    vals = [float(v) for v in (values or []) if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def sum_valid(values) -> float:
    vals = [float(v) for v in (values or []) if v is not None]
    return float(np.sum(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- #
def weather_for_event(key: str, lat: float, lon: float, date: str) -> dict:
    day = datetime.strptime(date, "%Y-%m-%d")
    start = (day - timedelta(days=ANTECEDENT_DAYS)).strftime("%Y-%m-%d")
    end = (day + timedelta(days=WINDOW_AFTER)).strftime("%Y-%m-%d")
    params = urllib.parse.urlencode({
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    })
    data = cached(f"wx_{key}", f"{ARCHIVE_URL}?{params}")

    daily = data.get("daily") or {}
    hourly = data.get("hourly") or {}
    d_dates = [datetime.strptime(t, "%Y-%m-%d") for t in daily.get("time", [])]
    h_times = [datetime.strptime(t, "%Y-%m-%dT%H:%M") for t in hourly.get("time", [])]

    w0 = day - timedelta(days=WINDOW_BEFORE)
    w1 = day + timedelta(days=WINDOW_AFTER)
    in_win = [w0 <= d <= w1 for d in d_dates]
    antecedent = [day - timedelta(days=ANTECEDENT_DAYS) <= d < day for d in d_dates]
    h_win = [w0 <= t < w1 + timedelta(days=1) for t in h_times]

    def pick(var, mask):
        return [v for v, m in zip(daily.get(var, []), mask) if m]

    soil = [v for v, m in zip(hourly.get("soil_moisture_0_to_7cm", []), h_win) if m]

    return {
        "precip_mean_mm_day": mean_valid(pick("precipitation_sum", in_win)),
        "precip_antecedent_30d_mm": sum_valid(pick("precipitation_sum", antecedent)),
        "temperature_mean_c": mean_valid(pick("temperature_2m_mean", in_win)),
        "rel_humidity_mean_pct": mean_valid(pick("relative_humidity_2m_mean", in_win)),
        "wind_max_mean_kmh": mean_valid(pick("wind_speed_10m_max", in_win)),
        "surface_pressure_mean_hpa": mean_valid(pick("surface_pressure_mean", in_win)),
        "et0_mean_mm_day": mean_valid(pick("et0_fao_evapotranspiration", in_win)),
        "soil_moisture_0_7cm_m3m3": mean_valid(soil),
        "openmeteo_grid_lat": data.get("latitude"),
        "openmeteo_grid_lon": data.get("longitude"),
        "openmeteo_grid_elevation_m": data.get("elevation"),
    }


def elevations(points: list[tuple[float, float]], tag: str) -> list[float]:
    """Open-Meteo elevation API (100 coordinates per request, paced for the free tier)."""
    out: list[float] = []
    for i in range(0, len(points), 100):
        chunk = points[i:i + 100]
        if not (CACHE / f"elev_{tag}_{i:05d}.json").exists():
            print(f"  elevation batch {i // 100 + 1}/{math.ceil(len(points) / 100)}")
            time.sleep(3.0)
        params = urllib.parse.urlencode({
            "latitude": ",".join(f"{p[0]:.5f}" for p in chunk),
            "longitude": ",".join(f"{p[1]:.5f}" for p in chunk),
        })
        data = cached(f"elev_{tag}_{i:05d}", f"{ELEVATION_URL}?{params}")
        out.extend(float(e) if e is not None else float("nan") for e in data["elevation"])
    return out


def slope_from_dem(df: pd.DataFrame) -> pd.DataFrame:
    d = SLOPE_OFFSET_DEG
    pts = []
    for lat, lon in zip(df.latitude, df.longitude):
        pts += [(lat, lon), (lat + d, lon), (lat - d, lon), (lat, lon + d), (lat, lon - d)]
    z = np.array(elevations(pts, "dem")).reshape(-1, 5)
    zc, zn, zs, ze, zw = z.T
    dy = 2 * d * EARTH_M_PER_DEG
    dx = 2 * d * EARTH_M_PER_DEG * np.cos(np.radians(df.latitude.to_numpy()))
    grad = np.sqrt(((zn - zs) / dy) ** 2 + ((ze - zw) / dx) ** 2)
    slope = np.degrees(np.arctan(grad))
    return pd.DataFrame({"elevation_m": zc, "slope_deg": slope}, index=df.index)


# --------------------------------------------------------------------------- #
def main() -> None:
    if not INPUT_CSV.exists():
        sys.exit(f"Input not found: {INPUT_CSV}")
    CACHE.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(INPUT_CSV, dtype={"source_id": str})
    meta = ["disaster_type", "latitude", "longitude", "event_date", "source", "source_id"]
    events = events[meta].copy()
    events["event_key"] = events.source + "_" + events.disaster_type + "_" + events.source_id
    print(f"Events: {len(events)}  ({events.event_key.nunique()} unique keys)")

    rows, failures = [], []
    for i, ev in enumerate(events.itertuples(index=False), 1):
        try:
            rows.append(weather_for_event(ev.event_key, ev.latitude, ev.longitude, ev.event_date))
        except Exception as err:  # noqa: BLE001
            failures.append(f"{ev.event_key}: {err}")
            rows.append({})
        if i % 25 == 0 or i == len(events):
            print(f"  weather {i}/{len(events)}  (failures: {len(failures)})")

    out = pd.concat([events.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    print("Fetching DEM elevation and slope ...")
    slope_ok = True
    try:
        out = pd.concat([out, slope_from_dem(out)], axis=1)
        out["over_water"] = (out.elevation_m <= 0).astype(int)
    except RuntimeError as err:
        slope_ok = False
        print("\n  Open-Meteo hourly limit reached during the elevation step.")
        print("  Weather features are saved; slope is left empty for now.")
        print("  Run this script again in about an hour to add slope (it resumes).")
        failures.append(f"elevation step: {err}")
        out["elevation_m"] = np.nan
        out["slope_deg"] = np.nan
        out["over_water"] = np.nan
    out.to_csv(OUTPUT_CSV, index=False)

    feats = [c for c in out.columns if c not in meta + ["event_key", "openmeteo_grid_lat",
                                                     "openmeteo_grid_lon"]]
    lines = [
        f"Created: {datetime.now():%Y-%m-%d %H:%M}",
        f"Slope complete: {slope_ok}",
        f"Input: {INPUT_CSV.name}  Events: {len(out)}",
        f"Weather window: event date -{WINDOW_BEFORE} to +{WINDOW_AFTER} days (daily means);"
        f" antecedent precipitation: {ANTECEDENT_DAYS} days before event (sum)",
        f"Daily vars: {', '.join(DAILY_VARS)}; hourly: {', '.join(HOURLY_VARS)}",
        f"Slope: central differences on Open-Meteo elevation (Copernicus GLO-90), offset {SLOPE_OFFSET_DEG} deg",
        "No noise added, no fallback constants, no class-dependent or catalogue-derived values.",
        "",
        "Missing values per feature:",
        out[feats].isna().sum().to_string(),
        "",
        "Summary:",
        out[feats].describe().T.round(3).to_string(),
        "",
        f"Failures ({len(failures)}):",
        *failures,
    ]
    LOG_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Wrote {OUTPUT_CSV}\n      and {LOG_FILE}")
    print("Please send both files back.")


if __name__ == "__main__":
    main()
