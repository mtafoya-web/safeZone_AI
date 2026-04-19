"""src/clean.py"""

from __future__ import annotations

import pandas as pd

from utils import PROCESSED_DIR, ensure_directories, get_logger


REQUIRED_COLUMNS = ["latitude", "longitude", "bright_ti4", "confidence", "acq_date"]
ESSENTIAL_COLUMNS = ["latitude", "longitude", "bright_ti4", "acq_date"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the core wildfire columns and only drop rows when essential fields are missing.

    A common reason for empty downstream CSVs is over-aggressive filtering here,
    especially if we drop on every column after reindexing missing fields.
    """
    logger = get_logger()
    logger.info("Cleaning data...")

    ensure_directories()
    logger.info(f"Rows before cleaning: {len(df)}")
    logger.info(f"Columns before cleaning: {list(df.columns)}")

    if df.empty:
        cleaned_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    else:
        available_columns = [column for column in REQUIRED_COLUMNS if column in df.columns]
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

        if missing_columns:
            logger.warning(f"Missing expected columns from FIRMS data: {missing_columns}")

        cleaned_df = df.reindex(columns=REQUIRED_COLUMNS).copy()
        cleaned_df = cleaned_df.dropna(subset=ESSENTIAL_COLUMNS)

        if "confidence" in cleaned_df.columns:
            cleaned_df["confidence"] = cleaned_df["confidence"].fillna("unknown")

    logger.info(f"Rows after cleaning: {len(cleaned_df)}")

    output_path = PROCESSED_DIR / "cleaned.csv"
    cleaned_df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")
    return cleaned_df
