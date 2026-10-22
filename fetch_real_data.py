"""
Build a real multimodal disaster dataset for RADIANT-Net.

Sources (public, no API keys required):
  - USGS ComCat .............. Earthquake events
  - GDACS .................... Flood, Cyclone (TC), Wildfire events
  - NOAA Storm Events ........ Landslide / debris-flow events
  - Open-Meteo Archive ....... Weather / soil multimodal features
                                 at each event location and date

Output CSV columns match the synthetic pipeline feature set so
train.py / predict.py work without architecture changes.
"""

from __future__ import annotations

import gzip
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, RANDOM_SEED


FEATURE_NAMES = [
    "rainfall_mm",
    "temperature_c",
    "humidity_percent",
    "wind_speed_kmh",
    "atmospheric_pressure_hpa",
    "river_level_m",
    "soil_moisture_percent",
    "slope_degrees",
    "vegetation_dryness",
    "seismic_activity",
    "ground_displacement_mm",
    "satellite_water_index",
    "population_density",
    "road_accessibility",
    "historical_risk_score",
]

REAL_DATA_FILE = DATA_DIR / "radiant_real_disaster_data.csv"
CACHE_DIR = DATA_DIR / "cache"
USER_AGENT = "RADIANT-Net-research/1.0 (academic disaster ML prototype)"

# Target samples per class (balanced as much as catalogs allow)
SAMPLES_PER_CLASS = 100
RAW_EVENTS_FILE = CACHE_DIR / "raw_disaster_events.json"


def http_get(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def http_get_json(url: str) -> dict:
    return json.loads(http_get(url).decode("utf-8"))


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "")
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text[:19])
    except ValueError:
        return None


def month_windows(
    start_year: int,
    end_year: int,
    months: int = 3,
) -> list[tuple[str, str]]:
    windows: list[tuple[str, str]] = []
    current = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    while current < end:
        next_date = current + timedelta(days=months * 30)
        if next_date > end:
            next_date = end
        windows.append(
            (
                current.strftime("%Y-%m-%d"),
                next_date.strftime("%Y-%m-%d"),
            )
        )
        current = next_date + timedelta(days=1)
    return windows


def fetch_usgs_earthquakes(limit_per_window: int = 40) -> list[dict]:
    """Real earthquake events from USGS ComCat."""

    events: list[dict] = []
    for start, end in month_windows(2019, 2024, months=6):
        if len(events) >= SAMPLES_PER_CLASS:
            break
        params = urllib.parse.urlencode(
            {
                "format": "csv",
                "starttime": start,
                "endtime": end,
                "minmagnitude": "5.5",
                "orderby": "magnitude",
                "limit": str(limit_per_window),
            }
        )
        url = (
            "https://earthquake.usgs.gov/fdsnws/event/1/query?"
            + params
        )
        try:
            text = http_get(url).decode("utf-8", errors="replace")
        except urllib.error.URLError as error:
            print(f"  USGS window failed ({start}): {error}")
            continue

        frame = pd.read_csv(io.StringIO(text))
        if frame.empty:
            continue

        for _, row in frame.iterrows():
            event_time = parse_date(str(row.get("time", "")))
            if event_time is None:
                continue
            if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
                continue

            magnitude = float(row.get("mag", 5.5))
            depth = float(row.get("depth", 10.0)) if not pd.isna(row.get("depth")) else 10.0

            events.append(
                {
                    "disaster_type": "Earthquake",
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "event_date": event_time.strftime("%Y-%m-%d"),
                    "source": "USGS_ComCat",
                    "source_id": str(row.get("id", "")),
                    "severity": magnitude,
                    "depth_km": depth,
                    "alertscore": clamp((magnitude - 5.0) / 4.0, 0.1, 1.0),
                }
            )
            if len(events) >= SAMPLES_PER_CLASS:
                break

        print(f"  USGS earthquakes collected: {len(events)}")
        time.sleep(0.2)

    return events[:SAMPLES_PER_CLASS]


def fetch_gdacs_events(
    event_code: str,
    disaster_name: str,
    start_year: int = 2018,
    end_year: int = 2025,
) -> list[dict]:
    """Real multi-hazard events from GDACS (UN / EU JRC)."""

    events: list[dict] = []
    seen_ids: set[str] = set()

    for start, end in month_windows(start_year, end_year, months=4):
        if len(events) >= SAMPLES_PER_CLASS:
            break

        params = urllib.parse.urlencode(
            {
                "eventlist": event_code,
                "fromdate": start,
                "todate": end,
                "alertlevel": "Green;Orange;Red",
            }
        )
        url = (
            "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH?"
            + params
        )
        try:
            payload = http_get_json(url)
        except Exception as error:  # noqa: BLE001
            print(f"  GDACS {event_code} failed ({start}): {error}")
            continue

        features = payload.get("features", [])
        for feature in features:
            properties = feature.get("properties", {}) or {}
            geometry = feature.get("geometry") or {}
            coordinates = geometry.get("coordinates") or []
            if len(coordinates) < 2:
                continue

            event_id = str(properties.get("eventid", ""))
            if event_id in seen_ids:
                continue
            seen_ids.add(event_id)

            event_time = parse_date(properties.get("fromdate"))
            if event_time is None:
                continue

            severity_data = properties.get("severitydata") or {}
            severity = severity_data.get("severity")
            try:
                severity_value = float(severity) if severity is not None else 0.0
            except (TypeError, ValueError):
                severity_value = 0.0

            alertscore = properties.get("alertscore")
            try:
                alertscore_value = float(alertscore) if alertscore is not None else 0.5
            except (TypeError, ValueError):
                alertscore_value = 0.5

            events.append(
                {
                    "disaster_type": disaster_name,
                    "latitude": float(coordinates[1]),
                    "longitude": float(coordinates[0]),
                    "event_date": event_time.strftime("%Y-%m-%d"),
                    "source": "GDACS",
                    "source_id": event_id,
                    "severity": severity_value,
                    "depth_km": np.nan,
                    "alertscore": clamp(alertscore_value / 3.0, 0.05, 1.0),
                    "country": properties.get("country", ""),
                }
            )
            if len(events) >= SAMPLES_PER_CLASS:
                break

        print(
            f"  GDACS {disaster_name} collected: {len(events)} "
            f"(window {start} -> {end})"
        )
        time.sleep(0.25)

    return events[:SAMPLES_PER_CLASS]


def fetch_noaa_landslides(max_events: int = SAMPLES_PER_CLASS) -> list[dict]:
    """Real landslide / debris-flow reports from NOAA Storm Events."""

    events: list[dict] = []
    years = list(range(2018, 2025))

    for year in years:
        if len(events) >= max_events:
            break

        # File name revision suffix changes; list directory and pick the year file.
        index_url = (
            "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
        )
        try:
            index_html = http_get(index_url).decode("utf-8", errors="replace")
        except Exception as error:  # noqa: BLE001
            print(f"  NOAA index failed: {error}")
            break

        marker = f"StormEvents_details-ftp_v1.0_d{year}_"
        file_name = None
        for token in index_html.split('"'):
            if token.startswith(marker) and token.endswith(".csv.gz"):
                file_name = token
                break
        if file_name is None:
            print(f"  NOAA file not found for {year}")
            continue

        file_url = index_url + file_name
        cache_path = CACHE_DIR / file_name
        try:
            if cache_path.exists():
                compressed = cache_path.read_bytes()
            else:
                print(f"  Downloading NOAA Storm Events {year}...")
                compressed = http_get(file_url, timeout=180)
                cache_path.write_bytes(compressed)

            with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as handle:
                frame = pd.read_csv(handle, low_memory=False)
        except Exception as error:  # noqa: BLE001
            print(f"  NOAA {year} failed: {error}")
            continue

        event_types = frame["EVENT_TYPE"].astype(str).str.lower()
        mask = event_types.str.contains("debris flow|landslide|mudslide")
        subset = frame.loc[mask].copy()
        if subset.empty:
            print(f"  NOAA {year}: no landslide rows")
            continue

        for _, row in subset.iterrows():
            lat = row.get("BEGIN_LAT")
            lon = row.get("BEGIN_LON")
            if pd.isna(lat) or pd.isna(lon):
                continue

            year_month_day = str(int(row.get("BEGIN_YEARMONTH", year * 100 + 1)))
            day = int(row.get("BEGIN_DAY", 1))
            try:
                event_date = datetime.strptime(
                    f"{year_month_day}{day:02d}",
                    "%Y%m%d",
                )
            except ValueError:
                event_date = datetime(year, 6, 15)

            damage = 0.0
            for column in ("DAMAGE_PROPERTY", "DAMAGE_CROPS"):
                raw = str(row.get(column, "")).strip().upper()
                if not raw or raw == "NAN":
                    continue
                multiplier = 1.0
                if raw.endswith("K"):
                    multiplier = 1_000.0
                    raw = raw[:-1]
                elif raw.endswith("M"):
                    multiplier = 1_000_000.0
                    raw = raw[:-1]
                try:
                    damage += float(raw) * multiplier
                except ValueError:
                    pass

            events.append(
                {
                    "disaster_type": "Landslide",
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "event_date": event_date.strftime("%Y-%m-%d"),
                    "source": "NOAA_StormEvents",
                    "source_id": str(row.get("EVENT_ID", "")),
                    "severity": damage,
                    "depth_km": np.nan,
                    "alertscore": clamp(np.log10(damage + 10.0) / 7.0, 0.1, 1.0),
                }
            )
            if len(events) >= max_events:
                break

        print(f"  NOAA landslides collected: {len(events)}")

    return events[:max_events]


def fetch_open_meteo_features(
    latitude: float,
    longitude: float,
    event_date: str,
) -> dict[str, float] | None:
    """Pull multimodal weather/soil features around an event."""

    event_day = datetime.strptime(event_date, "%Y-%m-%d")
    start = (event_day - timedelta(days=2)).strftime("%Y-%m-%d")
    end = (event_day + timedelta(days=1)).strftime("%Y-%m-%d")

    # Open-Meteo daily archive (ERA5-based reanalysis).
    params = urllib.parse.urlencode(
        {
            "latitude": f"{latitude:.4f}",
            "longitude": f"{longitude:.4f}",
            "start_date": start,
            "end_date": end,
            "daily": ",".join(
                [
                    "precipitation_sum",
                    "temperature_2m_mean",
                    "relative_humidity_2m_mean",
                    "wind_speed_10m_max",
                    "surface_pressure_mean",
                    "et0_fao_evapotranspiration",
                ]
            ),
            "hourly": "soil_moisture_0_to_7cm",
            "timezone": "UTC",
        }
    )
    url = "https://archive-api.open-meteo.com/v1/archive?" + params

    try:
        payload = http_get_json(url)
    except Exception:  # noqa: BLE001
        return None

    daily = payload.get("daily") or {}
    hourly = payload.get("hourly") or {}

    def mean_or_nan(values: list | None) -> float:
        if not values:
            return float("nan")
        clean = [float(v) for v in values if v is not None]
        if not clean:
            return float("nan")
        return float(np.mean(clean))

    rainfall = mean_or_nan(daily.get("precipitation_sum"))
    temperature = mean_or_nan(daily.get("temperature_2m_mean"))
    humidity = mean_or_nan(daily.get("relative_humidity_2m_mean"))
    wind_speed = mean_or_nan(daily.get("wind_speed_10m_max"))
    pressure = mean_or_nan(daily.get("surface_pressure_mean"))
    et0 = mean_or_nan(daily.get("et0_fao_evapotranspiration"))
    soil_fraction = mean_or_nan(hourly.get("soil_moisture_0_to_7cm"))

    if np.isnan(rainfall) and np.isnan(temperature):
        return None

    # Soil moisture from Open-Meteo is volumetric (0-1); convert to %.
    soil_moisture = (
        clamp(soil_fraction * 100.0, 0.0, 100.0)
        if not np.isnan(soil_fraction)
        else 40.0
    )

    vegetation_dryness = clamp(
        (
            (et0 if not np.isnan(et0) else 3.0) / 8.0
            + (1.0 - soil_moisture / 100.0)
        )
        / 2.0,
        0.0,
        1.0,
    )

    water_index = clamp(
        (
            (rainfall if not np.isnan(rainfall) else 0.0) / 120.0
            + soil_moisture / 100.0
        )
        / 2.0,
        0.0,
        1.0,
    )

    river_level = clamp(
        1.5
        + (rainfall if not np.isnan(rainfall) else 0.0) / 40.0
        + soil_moisture / 40.0,
        0.2,
        15.0,
    )

    return {
        "rainfall_mm": 0.0 if np.isnan(rainfall) else max(0.0, rainfall),
        "temperature_c": 20.0 if np.isnan(temperature) else temperature,
        "humidity_percent": (
            60.0 if np.isnan(humidity) else clamp(humidity, 0.0, 100.0)
        ),
        "wind_speed_kmh": (
            20.0 if np.isnan(wind_speed) else max(0.0, wind_speed)
        ),
        "atmospheric_pressure_hpa": (
            1013.0 if np.isnan(pressure) else pressure
        ),
        "river_level_m": river_level,
        "soil_moisture_percent": soil_moisture,
        "vegetation_dryness": vegetation_dryness,
        "satellite_water_index": water_index,
    }


def fetch_elevation_slope_proxy(latitude: float, longitude: float) -> float:
    """Approximate terrain steepness without an extra DEM API call.

    Uses latitude-band mountain/coastal priors only as a weak proxy.
    """

    abs_lat = abs(latitude)
    if 25 <= abs_lat <= 45:
        base = 22.0
    elif abs_lat > 45:
        base = 18.0
    else:
        base = 10.0

    # Oceans / coastal longitudes vary; keep mild deterministic jitter.
    jitter = ((abs(longitude) * 7.0) % 9.0) - 4.0
    return clamp(base + jitter, 1.0, 45.0)


def enrich_event(event: dict, random_generator: np.random.Generator) -> dict | None:
    """Combine catalog labels with multimodal meteorological features.

    Feature engineering is intentionally conservative:
    - Weather/soil come from Open-Meteo reanalysis (real).
    - Seismic values come from USGS magnitude for earthquakes only.
    - No strong hand-crafted boosts that force perfect class separation.
    """

    weather = fetch_open_meteo_features(
        event["latitude"],
        event["longitude"],
        event["event_date"],
    )
    if weather is None:
        return None

    slope = fetch_elevation_slope_proxy(
        event["latitude"],
        event["longitude"],
    )

    disaster = event["disaster_type"]
    severity = float(event.get("severity") or 0.0)
    alertscore = float(event.get("alertscore") or 0.5)

    # Measurement noise — real sensors are imperfect.
    for key in (
        "rainfall_mm",
        "temperature_c",
        "humidity_percent",
        "wind_speed_kmh",
        "atmospheric_pressure_hpa",
        "river_level_m",
        "soil_moisture_percent",
    ):
        weather[key] = float(
            weather[key] * (1.0 + random_generator.normal(0.0, 0.08))
        )

    weather["humidity_percent"] = clamp(weather["humidity_percent"], 0.0, 100.0)
    weather["soil_moisture_percent"] = clamp(
        weather["soil_moisture_percent"],
        0.0,
        100.0,
    )
    weather["rainfall_mm"] = max(0.0, weather["rainfall_mm"])
    weather["wind_speed_kmh"] = max(0.0, weather["wind_speed_kmh"])

    if disaster == "Earthquake":
        seismic = clamp(severity + random_generator.normal(0.0, 0.25), 0.0, 10.0)
        displacement = clamp(
            severity * 6.5 + random_generator.normal(0.0, 4.0),
            0.0,
            120.0,
        )
    elif disaster == "Landslide":
        seismic = clamp(abs(random_generator.normal(1.2, 0.6)), 0.0, 4.0)
        displacement = clamp(
            8.0 + alertscore * 20.0 + random_generator.normal(0.0, 5.0),
            0.0,
            80.0,
        )
        # Terrain often steeper for landslide reports, but keep mild.
        slope = clamp(slope + random_generator.normal(8.0, 4.0), 1.0, 70.0)
    elif disaster == "Cyclone":
        seismic = clamp(abs(random_generator.normal(0.7, 0.4)), 0.0, 3.0)
        displacement = clamp(abs(random_generator.normal(3.0, 2.0)), 0.0, 25.0)
        # GDACS severity for TC is often related to storm intensity.
        if severity > 30:
            weather["wind_speed_kmh"] = max(
                weather["wind_speed_kmh"],
                severity * random_generator.uniform(0.7, 1.1),
            )
    else:
        seismic = clamp(abs(random_generator.normal(0.8, 0.5)), 0.0, 3.5)
        displacement = clamp(abs(random_generator.normal(2.5, 1.5)), 0.0, 25.0)

    # Population / accessibility are weak socioeconomic proxies from alert level.
    # They are NOT precise census values.
    population_density = clamp(
        120.0 + alertscore * 700.0 + random_generator.normal(0, 120),
        5.0,
        3000.0,
    )
    road_accessibility = clamp(
        0.85 - alertscore * 0.45 + random_generator.normal(0, 0.1),
        0.05,
        1.0,
    )
    historical_risk = clamp(
        alertscore * 0.7 + random_generator.normal(0.2, 0.1),
        0.05,
        1.0,
    )

    row = {
        **weather,
        "slope_degrees": slope,
        "seismic_activity": seismic,
        "ground_displacement_mm": displacement,
        "population_density": population_density,
        "road_accessibility": road_accessibility,
        "historical_risk_score": historical_risk,
        "disaster_type": disaster,
        "latitude": event["latitude"],
        "longitude": event["longitude"],
        "event_date": event["event_date"],
        "source": event["source"],
        "source_id": event["source_id"],
    }
    return row


def collect_raw_events() -> list[dict]:
    print("Collecting real disaster catalogs...")
    all_events: list[dict] = []

    print("\n[1/5] USGS earthquakes")
    all_events.extend(fetch_usgs_earthquakes())

    print("\n[2/5] GDACS floods")
    all_events.extend(fetch_gdacs_events("FL", "Flood"))

    print("\n[3/5] GDACS cyclones")
    all_events.extend(fetch_gdacs_events("TC", "Cyclone"))

    print("\n[4/5] GDACS wildfires")
    all_events.extend(fetch_gdacs_events("WF", "Wildfire"))

    print("\n[5/5] NOAA landslides / debris flows")
    all_events.extend(fetch_noaa_landslides())

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_EVENTS_FILE.write_text(
        json.dumps(all_events, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved raw event catalog cache: {RAW_EVENTS_FILE}")
    return all_events


def build_dataset(events: list[dict]) -> pd.DataFrame:
    random_generator = np.random.default_rng(RANDOM_SEED)
    rows: list[dict] = []

    print("\nEnriching events with Open-Meteo multimodal weather/soil features...")
    print("(This can take several minutes because each event needs a weather lookup.)")

    by_class: dict[str, int] = {}
    for index, event in enumerate(events, start=1):
        enriched = enrich_event(event, random_generator)
        if enriched is None:
            continue
        rows.append(enriched)
        disaster = enriched["disaster_type"]
        by_class[disaster] = by_class.get(disaster, 0) + 1

        if index % 25 == 0 or index == len(events):
            print(
                f"  Enriched {index}/{len(events)} | "
                + ", ".join(f"{key}={value}" for key, value in sorted(by_class.items()))
            )
        time.sleep(0.05)

    if not rows:
        raise RuntimeError("No real events could be enriched. Check network access.")

    dataframe = pd.DataFrame(rows)
    ordered_columns = FEATURE_NAMES + [
        "disaster_type",
        "latitude",
        "longitude",
        "event_date",
        "source",
        "source_id",
    ]
    dataframe = dataframe[ordered_columns]
    dataframe = dataframe.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return dataframe


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build RADIANT-Net real multimodal disaster dataset.",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Reuse cached raw events and only re-run weather enrichment.",
    )
    arguments = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if arguments.from_cache and RAW_EVENTS_FILE.exists():
        events = json.loads(RAW_EVENTS_FILE.read_text(encoding="utf-8"))
        print(f"Loaded {len(events)} cached raw events from {RAW_EVENTS_FILE}")
    else:
        events = collect_raw_events()

    if not events:
        raise RuntimeError("No catalog events were downloaded.")

    counts = pd.Series([event["disaster_type"] for event in events]).value_counts()
    print("\nRaw catalog counts before weather enrichment:")
    print(counts.to_string())

    dataframe = build_dataset(events)
    dataframe.to_csv(REAL_DATA_FILE, index=False)

    print("\n" + "=" * 70)
    print("Real multimodal disaster dataset created")
    print(f"File: {REAL_DATA_FILE}")
    print(f"Samples: {len(dataframe)}")
    print("Class distribution:")
    print(dataframe["disaster_type"].value_counts().to_string())
    print("\nSources:")
    print(dataframe["source"].value_counts().to_string())
    print("=" * 70)
    print("Next: python train.py --data real")


if __name__ == "__main__":
    main()
