"""src/data_fetch.py"""

from __future__ import annotations

import concurrent.futures
import io
import os
from dataclasses import dataclass

import pandas as pd
import requests
from requests import Response

from utils import RAW_DIR, ensure_directories, get_logger, load_environment


FIRMS_SOURCE = "VIIRS_NOAA20_NRT"
CALIFORNIA_BBOX = "-124.48,32.53,-114.13,42.01"
FIRMS_URL_TEMPLATE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source}/{bbox}/{days}"
FIRMS_TIMEOUT = 30
FIRMS_WINDOWS = (30, 14, 7, 3)
DEDUP_COLUMNS = [
    "latitude",
    "longitude",
    "bright_ti4",
    "acq_date",
    "acq_time",
    "satellite",
]


@dataclass
class FirmsFetchResult:
    """Container for one FIRMS window attempt."""

    days: int
    dataframe: pd.DataFrame | None = None
    status_code: int | None = None
    error: str | None = None


def _response_looks_like_csv(response_text: str) -> bool:
    """
    Check whether the NASA response looks like CSV instead of an HTML or error page.

    A common failure mode is getting a plain-text or HTML error back and passing it
    into pandas, which then produces a confusing empty dataframe.
    """
    if not response_text or not response_text.strip():
        return False

    first_line = response_text.strip().splitlines()[0].lower()
    return "," in first_line and "latitude" in first_line and "longitude" in first_line


def _load_csv_from_response(response: Response) -> pd.DataFrame:
    """Load NASA CSV text into pandas after validating the format."""
    response_text = response.text.strip()

    if not response_text:
        raise ValueError("NASA FIRMS returned an empty response body.")

    if not _response_looks_like_csv(response_text):
        raise ValueError(
            "NASA FIRMS did not return valid CSV. "
            "This usually means the API key is invalid, the URL is wrong, or the service returned an error page."
        )

    df = pd.read_csv(io.StringIO(response.text))
    return df


def _deduplicate_firms_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate FIRMS rows to keep CSV outputs stable."""
    available_columns = [column for column in DEDUP_COLUMNS if column in df.columns]
    if not available_columns or df.empty:
        return df

    return df.drop_duplicates(subset=available_columns).reset_index(drop=True)


def _request_firms_window(api_key: str, days: int) -> FirmsFetchResult:
    """
    Fetch one FIRMS time window.

    The FIRMS API returns a complete CSV for a given window, so there is no chunked
    pagination work to parallelize inside a single request. The useful concurrency
    here is running the fallback windows in parallel so we do not wait for a second
    request only after the first one comes back empty.
    """
    try:
        url = FIRMS_URL_TEMPLATE.format(
            api_key=api_key,
            source=FIRMS_SOURCE,
            bbox=CALIFORNIA_BBOX,
            days=days,
        )
        response = requests.get(url, timeout=FIRMS_TIMEOUT)
        response.raise_for_status()
        dataframe = _load_csv_from_response(response)
        dataframe = _deduplicate_firms_rows(dataframe)

        return FirmsFetchResult(days=days, dataframe=dataframe, status_code=response.status_code)
    except requests.RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return FirmsFetchResult(
            days=days,
            status_code=status_code,
            error=(
                f"NASA FIRMS request failed with status {status_code or 'unknown'}. "
                "Check the API key, request URL, or NASA service availability."
            ),
        )
    except Exception as exc:
        return FirmsFetchResult(days=days, error=str(exc))


def fetch_firms_data() -> pd.DataFrame:
    """
    Fetch NASA FIRMS fire detections for California and save the raw CSV locally.

    We keep the existing preference for a shorter recent window first, but we run the
    3-day and 7-day requests in parallel because they are independent I/O-bound calls.
    This reduces wait time when the 3-day window is empty and we need the 7-day data.
    """
    logger = get_logger()
    logger.info("Fetching FIRMS data...")

    load_environment()
    ensure_directories()

    nasa_api_key = os.getenv("NASA_API_KEY", "").strip()
    logger.info(f"NASA_API_KEY found: {bool(nasa_api_key)}")

    if not nasa_api_key:
        raise EnvironmentError(
            "NASA_API_KEY was not found in .env. Add it to the project root .env file before running the pipeline."
        )

    output_path = RAW_DIR / "firms.csv"
    results_by_days: dict[int, FirmsFetchResult] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(FIRMS_WINDOWS)) as executor:
        future_to_days = {
            executor.submit(_request_firms_window, nasa_api_key, days): days for days in FIRMS_WINDOWS
        }

        for future in concurrent.futures.as_completed(future_to_days):
            result = future.result()
            results_by_days[result.days] = result
            logger.info(f"FIRMS window requested: last {result.days} day(s)")
            logger.info(f"FIRMS response status code for {result.days}-day window: {result.status_code or 'error'}")

            if result.error:
                logger.warning(f"FIRMS request for {result.days}-day window failed: {result.error}")
            else:
                logger.info(
                    f"Rows returned for {result.days}-day window: {len(result.dataframe) if result.dataframe is not None else 0}"
                )

    selected_df: pd.DataFrame | None = None

    for days in sorted(FIRMS_WINDOWS,reverse=True):
        result = results_by_days.get(days)
        if result is None:
            continue
        if result.dataframe is not None and not result.dataframe.empty:
            selected_df = result.dataframe
            logger.info(f"Using FIRMS data from the {days}-day window.")
            break

    if selected_df is None:
        errors = [result.error for result in results_by_days.values() if result.error]
        if errors:
            raise RuntimeError(
                "FIRMS data fetch failed after trying the configured windows. "
                f"Last error: {errors[-1]}"
            )

        raise RuntimeError(
            "NASA FIRMS returned valid responses for both 3-day and 7-day windows, "
            "but no California rows were found."
        )

    selected_df = _deduplicate_firms_rows(selected_df)
    logger.info(f"Final deduplicated FIRMS row count: {len(selected_df)}")
    if "acq_date" in selected_df.columns:
        logger.info(f"Unique acq_date count: {selected_df['acq_date'].nunique()}")
        logger.info(f"Date range: {selected_df['acq_date'].min()} -> {selected_df['acq_date'].max()}")
    selected_df.to_csv(output_path, index=False)
    logger.info(f"Saved raw FIRMS data to {output_path}")
    return selected_df
