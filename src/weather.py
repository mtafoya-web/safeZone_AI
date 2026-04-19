"""src/weather.py"""

from __future__ import annotations

import concurrent.futures
import os
import threading
import time
from typing import Dict, Tuple

import pandas as pd

from utils import get_env_int, get_logger, parse_json_response, parse_wind_speed, safe_request


USER_AGENT = "safeZone-AI/1.0 (educational geospatial ML pipeline)"
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/geo+json, application/json",
}

_WEATHER_CACHE: Dict[Tuple[float, float], dict[str, int]] = {}
_CACHE_LOCK = threading.Lock()
VERBOSE_WEATHER_LOGS = os.getenv("WEATHER_VERBOSE_LOGS", "").lower() in {"1", "true", "yes"}
WEATHER_TIMEOUT = get_env_int("WEATHER_REQUEST_TIMEOUT", 20)
WEATHER_BIN_PRECISION = get_env_int("WEATHER_BIN_PRECISION", 2)
WEATHER_MAX_WORKERS = max(1, get_env_int("WEATHER_MAX_WORKERS", min(8, (os.cpu_count() or 4) * 2)))
WEATHER_REQUEST_DELAY_SECONDS = max(0.0, float(os.getenv("WEATHER_REQUEST_DELAY_SECONDS", "0")))


def _debug_log(message: str) -> None:
    """Print NOAA details only when verbose weather logging is enabled."""
    if VERBOSE_WEATHER_LOGS:
        get_logger().info(message)


def _round_weather_bin(lat: float, lon: float) -> tuple[float, float]:
    """Round coordinates so nearby points share the same NOAA lookup."""
    return (round(float(lat), WEATHER_BIN_PRECISION), round(float(lon), WEATHER_BIN_PRECISION))


def _default_weather_result() -> dict[str, int]:
    """Return a safe fallback when NOAA data is unavailable."""
    return {"wind_speed": 0, "alert_flag": 0}


def _read_cached_weather(cache_key: tuple[float, float]) -> dict[str, int] | None:
    """Read from the shared weather cache in a thread-safe way."""
    with _CACHE_LOCK:
        cached_value = _WEATHER_CACHE.get(cache_key)
        return dict(cached_value) if cached_value is not None else None


def _write_cached_weather(cache_key: tuple[float, float], weather_result: dict[str, int]) -> None:
    """Write to the shared weather cache in a thread-safe way."""
    with _CACHE_LOCK:
        _WEATHER_CACHE[cache_key] = dict(weather_result)


def _fetch_weather_for_bin(cache_key: tuple[float, float]) -> dict[str, int]:
    """
    Fetch weather for one unique rounded bin.

    We use threads instead of multiprocessing because this work is dominated by
    network wait time, not CPU-heavy computation.
    """
    cached_value = _read_cached_weather(cache_key)
    if cached_value is not None:
        get_logger().info(f"Using cached weather for bin ({cache_key[0]}, {cache_key[1]})")
        return cached_value

    logger = get_logger()
    logger.info(f"Fetching weather for bin ({cache_key[0]}, {cache_key[1]})")

    if WEATHER_REQUEST_DELAY_SECONDS > 0:
        time.sleep(WEATHER_REQUEST_DELAY_SECONDS)

    points_url = f"https://api.weather.gov/points/{cache_key[0]},{cache_key[1]}"
    _debug_log(f"NOAA points URL: {points_url}")
    points_response = safe_request(points_url, headers=REQUEST_HEADERS, timeout=WEATHER_TIMEOUT)
    points_data = parse_json_response(points_response)

    forecast_url = points_data.get("properties", {}).get("forecast")
    if not forecast_url:
        logger.warning(
            "Forecast URL was missing from NOAA points response. "
            "Using default weather values so we do not drop rows."
        )
        default_result = _default_weather_result()
        _write_cached_weather(cache_key, default_result)
        return default_result

    _debug_log(f"NOAA forecast URL: {forecast_url}")
    forecast_response = safe_request(forecast_url, headers=REQUEST_HEADERS, timeout=WEATHER_TIMEOUT)
    forecast_data = parse_json_response(forecast_response)
    periods = forecast_data.get("properties", {}).get("periods", [])
    first_period = periods[0] if periods else {}
    wind_speed = parse_wind_speed(first_period.get("windSpeed"))

    alerts_url = "https://api.weather.gov/alerts/active"
    _debug_log(f"NOAA alerts URL: {alerts_url}?point={cache_key[0]},{cache_key[1]}")
    alerts_response = safe_request(
        alerts_url,
        params={"point": f"{cache_key[0]},{cache_key[1]}"},
        headers=REQUEST_HEADERS,
        timeout=WEATHER_TIMEOUT,
    )
    alerts_data = parse_json_response(alerts_response)
    features = alerts_data.get("features", [])
    alert_flag = int(
        any(feature.get("properties", {}).get("event") == "Red Flag Warning" for feature in features)
    )

    result = {"wind_speed": wind_speed, "alert_flag": alert_flag}
    _debug_log(
        f"Weather result for ({cache_key[0]}, {cache_key[1]}): "
        f"wind_speed={wind_speed}, alert_flag={alert_flag}"
    )
    _write_cached_weather(cache_key, result)
    return result


def get_weather(lat: float, lon: float) -> dict[str, int]:
    """Backward-compatible wrapper for fetching weather for one point."""
    return _fetch_weather_for_bin(_round_weather_bin(lat, lon))


def enrich_weather_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a dataframe with NOAA weather data using concurrent requests.

    We deduplicate rounded bins first so duplicate fire points share one NOAA call.
    That is where most of the speedup comes from.
    """
    logger = get_logger()
    weather_df = df.copy()

    if weather_df.empty:
        weather_df["wind_speed"] = pd.Series(dtype="int64")
        weather_df["alert_flag"] = pd.Series(dtype="int64")
        return weather_df

    weather_df["weather_lat_bin"] = weather_df["latitude"].round(WEATHER_BIN_PRECISION)
    weather_df["weather_lon_bin"] = weather_df["longitude"].round(WEATHER_BIN_PRECISION)

    unique_bins = (
        weather_df[["weather_lat_bin", "weather_lon_bin"]]
        .drop_duplicates()
        .rename(columns={"weather_lat_bin": "lat_bin", "weather_lon_bin": "lon_bin"})
        .reset_index(drop=True)
    )

    unique_bin_pairs = [tuple(row) for row in unique_bins[["lat_bin", "lon_bin"]].itertuples(index=False, name=None)]
    worker_count = min(WEATHER_MAX_WORKERS, max(1, len(unique_bin_pairs)))
    logger.info(f"Enriching weather for {len(unique_bin_pairs)} unique bins with {worker_count} workers...")

    weather_lookup: dict[tuple[float, float], dict[str, int]] = {}
    processed_bins = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_bin = {
            executor.submit(_fetch_weather_for_bin, bin_key): bin_key for bin_key in unique_bin_pairs
        }

        for future in concurrent.futures.as_completed(future_to_bin):
            bin_key = future_to_bin[future]
            try:
                weather_lookup[bin_key] = future.result()
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    f"Weather fetch failed unexpectedly for bin ({bin_key[0]}, {bin_key[1]}): {exc}. "
                    "Using safe fallback values."
                )
                weather_lookup[bin_key] = _default_weather_result()

            processed_bins += 1
            if processed_bins % 10 == 0 or processed_bins == len(unique_bin_pairs):
                logger.info(f"Processed weather for {processed_bins} of {len(unique_bin_pairs)} unique bins...")

    weather_lookup_df = unique_bins.copy()
    weather_lookup_df["weather_result"] = [
        weather_lookup.get((row.lat_bin, row.lon_bin), _default_weather_result())
        for row in weather_lookup_df.itertuples(index=False)
    ]
    weather_lookup_df["wind_speed"] = weather_lookup_df["weather_result"].apply(lambda item: item["wind_speed"])
    weather_lookup_df["alert_flag"] = weather_lookup_df["weather_result"].apply(lambda item: item["alert_flag"])
    weather_lookup_df = weather_lookup_df.drop(columns=["weather_result"])
    weather_lookup_df = weather_lookup_df.rename(
        columns={"lat_bin": "weather_lat_bin", "lon_bin": "weather_lon_bin"}
    )

    enriched_df = weather_df.merge(
        weather_lookup_df,
        on=["weather_lat_bin", "weather_lon_bin"],
        how="left",
    )
    enriched_df["wind_speed"] = pd.to_numeric(enriched_df["wind_speed"], errors="coerce").fillna(0).astype(int)
    enriched_df["alert_flag"] = pd.to_numeric(enriched_df["alert_flag"], errors="coerce").fillna(0).astype(int)

    return enriched_df.drop(columns=["weather_lat_bin", "weather_lon_bin"])
