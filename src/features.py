"""src/features.py"""

from __future__ import annotations

import pandas as pd

from utils import PROCESSED_DIR, ensure_directories, get_logger
from weather import enrich_weather_dataframe


CONFIDENCE_MAP = {
    "l": 25,
    "low": 25,
    "n": 60,
    "nominal": 60,
    "h": 90,
    "high": 90,
}


def _normalize_confidence(confidence_series: pd.Series) -> pd.Series:
    """Convert FIRMS confidence values into a numeric score when possible."""
    numeric_confidence = pd.to_numeric(confidence_series, errors="coerce")
    text_confidence = confidence_series.astype(str).str.strip().str.lower().map(CONFIDENCE_MAP)
    return numeric_confidence.fillna(text_confidence).fillna(0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert fire detections into one row per spatial bin per day.

    This is the key modeling shift that turns the project from rule imitation into
    a small forecasting problem. Each row now describes the state of one spatial
    bin on one calendar day.
    """
    logger = get_logger()
    logger.info("Engineering bin-level daily features...")

    featured_df = df.copy()
    if featured_df.empty:
        return pd.DataFrame(
            columns=[
                "acq_date",
                "lat_bin",
                "lon_bin",
                "latitude",
                "longitude",
                "fire_count",
                "avg_brightness",
                "max_brightness",
                "avg_confidence_score",
                "prev_day_fire_count",
                "prev_day_avg_brightness",
                "rolling_2day_fire_count",
                "rolling_3day_fire_count",
                "next_day_fire_count",
                "fire_next_day",
                "escalates_next_day",
            ]
        )

    featured_df["acq_date"] = pd.to_datetime(featured_df["acq_date"], errors="coerce")
    featured_df = featured_df.dropna(subset=["acq_date"]).copy()
    featured_df["confidence_score"] = _normalize_confidence(featured_df["confidence"])
    featured_df["lat_bin"] = featured_df["latitude"].round(1)
    featured_df["lon_bin"] = featured_df["longitude"].round(1)

    grouped_df = (
        featured_df.groupby(["acq_date", "lat_bin", "lon_bin"], as_index=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            fire_count=("latitude", "size"),
            avg_brightness=("bright_ti4", "mean"),
            max_brightness=("bright_ti4", "max"),
            avg_confidence_score=("confidence_score", "mean"),
        )
        .sort_values(["lat_bin", "lon_bin", "acq_date"])
        .reset_index(drop=True)
    )

    bin_groups = grouped_df.groupby(["lat_bin", "lon_bin"], group_keys=False)
    grouped_df["prev_day_fire_count"] = bin_groups["fire_count"].shift(1).fillna(0)
    grouped_df["prev_day_avg_brightness"] = bin_groups["avg_brightness"].shift(1).fillna(0)
    grouped_df["rolling_2day_fire_count"] = (
        bin_groups["fire_count"].rolling(window=2, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    )
    grouped_df["rolling_3day_fire_count"] = (
        bin_groups["fire_count"].rolling(window=3, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    )

    grouped_df["next_observed_date"] = bin_groups["acq_date"].shift(-1)
    grouped_df["next_observed_fire_count"] = bin_groups["fire_count"].shift(-1)
    grouped_df["expected_next_date"] = grouped_df["acq_date"] + pd.Timedelta(days=1)

    has_true_next_day = grouped_df["next_observed_date"] == grouped_df["expected_next_date"]
    grouped_df["next_day_fire_count"] = grouped_df["next_observed_fire_count"].where(has_true_next_day, 0).fillna(0)
    grouped_df["fire_next_day"] = (grouped_df["next_day_fire_count"] > 0).astype(int)
    grouped_df["escalates_next_day"] = (grouped_df["next_day_fire_count"] > grouped_df["fire_count"]).astype(int)

    max_observed_date = grouped_df["acq_date"].max()
    grouped_df = grouped_df[grouped_df["acq_date"] < max_observed_date].copy()
    logger.info(
        "Dropped the final observed date because we cannot know its next-day outcome from the current dataset."
    )

    grouped_df = grouped_df.drop(columns=["next_observed_date", "next_observed_fire_count", "expected_next_date"])
    logger.info(f"Rows after bin-day feature engineering: {len(grouped_df)}")
    return grouped_df


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Add NOAA weather features to each bin-day row."""
    logger = get_logger()
    logger.info("Adding weather features...")

    weather_df = df.copy()
    if weather_df.empty:
        weather_df["wind_speed"] = pd.Series(dtype="int64")
        weather_df["alert_flag"] = pd.Series(dtype="int64")
        logger.info("Weather enrichment received an empty dataframe.")
        return weather_df

    enriched_df = enrich_weather_dataframe(weather_df)
    logger.info(f"Rows after weather enrichment: {len(enriched_df)}")
    return enriched_df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep a transparent rule-based current-day risk label alongside ML forecasting.

    The model no longer trains on this label. It stays in the dataset as an
    explainable summary for demos and portfolio storytelling.
    """
    logger = get_logger()
    logger.info("Creating rule-based current-day risk labels...")

    labeled_df = df.copy()
    if labeled_df.empty:
        labeled_df["risk_level"] = pd.Series(dtype="object")
        logger.info("Label creation received an empty dataframe.")
        return labeled_df

    def classify_risk(row: pd.Series) -> str:
        if row["max_brightness"] > 330 or row["fire_count"] > 5 or row["wind_speed"] > 20:
            return "High"
        if row["avg_brightness"] > 300 or row["fire_count"] > 2:
            return "Medium"
        return "Low"

    labeled_df["risk_level"] = labeled_df.apply(classify_risk, axis=1)

    ensure_directories()
    output_path = PROCESSED_DIR / "bin_daily_features.csv"
    labeled_df.to_csv(output_path, index=False)
    logger.info(f"Saved bin-day features to {output_path}")
    logger.info(f"Rows after label creation: {len(labeled_df)}")
    return labeled_df
