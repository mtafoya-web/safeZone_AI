"""src/data_fetch_arcgis.py"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from utils import RAW_DIR, ensure_directories, get_logger


ARCGIS_HISTORICAL_URL = (
    "https://apps.fs.usda.gov/arcx/rest/services/"
    "EDW/EDW_FireOccurrenceAndPerimeter_01/MapServer/9/query"
)

ARCGIS_TIMEOUT = 30
ARCGIS_PAGE_SIZE = 2000

# California-ish bounding box to keep the dataset relevant to your project.
CA_MIN_LAT = 32.53
CA_MAX_LAT = 42.01
CA_MIN_LON = -124.48
CA_MAX_LON = -114.13

# Keep the historical pull manageable. Adjust this if you want more history.
MIN_FIRE_YEAR = 2020

DEDUP_COLUMNS = ["latitude", "longitude", "acq_date"]


def _build_where_clause() -> str:
    """Build a simple SQL-style filter for California-like bounds and recent years."""
    return (
        f"FIREYEAR >= {MIN_FIRE_YEAR} "
        f"AND LATDD83 >= {CA_MIN_LAT} AND LATDD83 <= {CA_MAX_LAT} "
        f"AND LONGDD83 >= {CA_MIN_LON} AND LONGDD83 <= {CA_MAX_LON}"
    )


def _fetch_arcgis_page(offset: int) -> dict:
    """Fetch one page of ArcGIS historical fire occurrence points."""
    params = {
        "where": _build_where_clause(),
        "outFields": "DISCOVERYDATETIME,LATDD83,LONGDD83,FIREYEAR,TOTALACRES,FIRENAME",
        "returnGeometry": "false",
        "f": "json",
        "resultOffset": offset,
        "resultRecordCount": ARCGIS_PAGE_SIZE,
        "orderByFields": "DISCOVERYDATETIME ASC",
    }

    response = requests.get(ARCGIS_HISTORICAL_URL, params=params, timeout=ARCGIS_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _normalize_arcgis_records(features: list[dict]) -> list[dict]:
    """Map ArcGIS attributes into the pipeline's expected raw schema."""
    records: list[dict] = []

    for feature in features:
        attrs = feature.get("attributes", {}) or {}

        lat = attrs.get("LATDD83")
        lon = attrs.get("LONGDD83")
        raw_date = attrs.get("DISCOVERYDATETIME")
        total_acres = attrs.get("TOTALACRES")

        if lat is None or lon is None or raw_date is None:
            continue

        # ArcGIS date fields are commonly epoch milliseconds.
        acq_date = pd.to_datetime(raw_date, unit="ms", errors="coerce")
        if pd.isna(acq_date):
            acq_date = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(acq_date):
            continue

        # This source does not provide FIRMS brightness. We use a simple acreage-based
        # proxy so the existing cleaning/features pipeline can still run.
        acreage = 0.0 if total_acres is None else float(total_acres)
        brightness_proxy = min(300.0 + acreage * 0.05, 360.0)

        records.append(
            {
                "latitude": float(lat),
                "longitude": float(lon),
                "bright_ti4": float(brightness_proxy),
                "confidence": "n",
                "acq_date": acq_date.date().isoformat(),
            }
        )

    return records


def _deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate historical rows."""
    if df.empty:
        return df
    available = [col for col in DEDUP_COLUMNS if col in df.columns]
    if not available:
        return df
    return df.drop_duplicates(subset=available).reset_index(drop=True)


def fetch_arcgis_historical_data() -> pd.DataFrame:
    """
    Fetch historical wildfire occurrence points from ArcGIS and save them locally.

    This is intended as historical training data to complement recent FIRMS data.
    """
    logger = get_logger()
    logger.info("Fetching ArcGIS historical wildfire data...")

    ensure_directories()

    all_records: list[dict] = []
    offset = 0

    while True:
        data = _fetch_arcgis_page(offset=offset)

        if "error" in data:
            raise RuntimeError(f"ArcGIS historical query failed: {data['error']}")

        features = data.get("features", [])
        if not features:
            break

        page_records = _normalize_arcgis_records(features)
        all_records.extend(page_records)

        logger.info(
            f"Fetched ArcGIS page at offset {offset}: "
            f"{len(features)} raw features, {len(page_records)} usable records."
        )

        if len(features) < ARCGIS_PAGE_SIZE:
            break

        offset += ARCGIS_PAGE_SIZE

    df = pd.DataFrame(all_records)
    df = _deduplicate_rows(df)

    if df.empty:
        logger.warning("No ArcGIS historical features returned after filtering.")
        return df

    output_path = RAW_DIR / "arcgis_history.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"ArcGIS historical rows fetched: {len(df)}")
    logger.info(f"ArcGIS historical unique dates: {df['acq_date'].nunique()}")
    logger.info(f"ArcGIS historical date range: {df['acq_date'].min()} -> {df['acq_date'].max()}")
    logger.info(f"Saved ArcGIS historical data to {output_path}")

    return df