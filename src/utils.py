"""src/utils.py"""

from __future__ import annotations

import io
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"


def ensure_directories() -> None:
    """Create the expected project folders if they do not exist yet."""
    for directory in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_environment() -> None:
    """Load environment variables from the project-level .env file."""
    load_dotenv(PROJECT_ROOT / ".env")


def get_logger(name: str = "wildfire_pipeline") -> logging.Logger:
    """Create a simple console logger for progress messages."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def get_env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a safe fallback."""
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    try:
        return int(raw_value)
    except ValueError:
        get_logger().warning(f"Invalid integer for {name}: {raw_value!r}. Using default {default}.")
        return default


def safe_request(
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: int = 30,
) -> Optional[requests.Response]:
    """
    Make an HTTP request and return None instead of raising on failure.

    This keeps the pipeline moving even if an external API is temporarily down.
    """
    logger = get_logger()

    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        logger.warning(f"Request failed for {url}: {exc}")
        return None


def parse_csv_response(response: Optional[requests.Response]) -> pd.DataFrame:
    """Convert a CSV response into a DataFrame, or return an empty frame on failure."""
    if response is None:
        return pd.DataFrame()

    try:
        return pd.read_csv(io.StringIO(response.text))
    except Exception as exc:  # pragma: no cover - defensive fallback
        get_logger().warning(f"Could not read CSV response: {exc}")
        return pd.DataFrame()


def parse_json_response(response: Optional[requests.Response]) -> dict[str, Any]:
    """Convert a JSON response into a dictionary, or return an empty one on failure."""
    if response is None:
        return {}

    try:
        return response.json()
    except ValueError as exc:
        get_logger().warning(f"Could not decode JSON response: {exc}")
        return {}


def parse_wind_speed(wind_speed_text: Any) -> int:
    """
    Convert NOAA wind strings like '10 mph' or '5 to 10 mph' into an integer.

    When a range is provided, we use the highest value because it is more useful
    for risk scoring.
    """
    if wind_speed_text is None:
        return 0

    numbers = [int(value) for value in re.findall(r"\d+", str(wind_speed_text))]
    return max(numbers) if numbers else 0
