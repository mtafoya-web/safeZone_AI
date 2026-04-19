"""src/dashboard.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path("output/predictions.csv")
OUTPUT_PATH = Path("output/fire_data.csv")

# Dashboard controls
LOOKBACK_DAYS = 1  # latest day + previous day
ONLY_ACTIONABLE_ROWS = False  # True -> keep only prepare / urgent_review
ONLY_FORECAST_ROWS = False  # True -> keep only predicted next-day active rows

DASHBOARD_COLUMNS = [
    "acq_date",
    "lat_bin",
    "lon_bin",
    "latitude",
    "longitude",
    "fire_count",
    "avg_brightness",
    "max_brightness",
    "wind_speed",
    "alert_flag",
    "risk_level",
    "probability",
    "predicted_fire_next_day",
    "advisory_score",
    "advisory_level",
    "review_flag",
    "is_forecast",
]


def create_dashboard_dataset() -> None:
    print("Creating dashboard dataset...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    if df.empty:
        print("predictions.csv is empty. No dashboard dataset created.")
        return

    # Ensure datetime
    if "acq_date" not in df.columns:
        raise KeyError("Column 'acq_date' is missing from predictions.csv")

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date"]).copy()

    if df.empty:
        print("No valid acq_date values found. No dashboard dataset created.")
        return

    # Keep latest day + previous day
    latest = df["acq_date"].max()
    cutoff = latest - pd.Timedelta(days=LOOKBACK_DAYS)
    df = df[df["acq_date"] >= cutoff].copy()

    # Rename probability column if needed
    if "probability" not in df.columns and "predicted_fire_next_day_probability" in df.columns:
        df = df.rename(columns={"predicted_fire_next_day_probability": "probability"})

    # Ensure ArcGIS-friendly coordinate names
    if "latitude" not in df.columns and "lat_bin" in df.columns:
        df["latitude"] = df["lat_bin"]

    if "longitude" not in df.columns and "lon_bin" in df.columns:
        df["longitude"] = df["lon_bin"]

    # Add forecast flag if missing
    if "is_forecast" not in df.columns:
        if "predicted_fire_next_day" in df.columns:
            df["is_forecast"] = (pd.to_numeric(df["predicted_fire_next_day"], errors="coerce").fillna(0) == 1).astype(int)
        else:
            df["is_forecast"] = 0

    # Optional filters for cleaner national dashboards
    if ONLY_ACTIONABLE_ROWS and "advisory_level" in df.columns:
        df = df[df["advisory_level"].isin(["prepare", "urgent_review"])].copy()

    if ONLY_FORECAST_ROWS and "predicted_fire_next_day" in df.columns:
        predicted = pd.to_numeric(df["predicted_fire_next_day"], errors="coerce").fillna(0)
        df = df[predicted == 1].copy()

    # Keep only useful dashboard columns
    available_columns = [col for col in DASHBOARD_COLUMNS if col in df.columns]
    df = df[available_columns].copy()

    # Drop anything missing coordinates
    required_coordinate_columns = ["latitude", "longitude"]
    missing_coordinate_columns = [col for col in required_coordinate_columns if col not in df.columns]
    if missing_coordinate_columns:
        raise KeyError(
            f"Missing coordinate columns required for dashboard mapping: {missing_coordinate_columns}"
        )

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["forecast_label"] = df["predicted_fire_next_day"].map({
        0: "No",
        1: "Yes"
    })
    df = df.dropna(subset=["latitude", "longitude"]).copy()

    # Save
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved dashboard dataset to {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Dates: {df['acq_date'].min()} -> {df['acq_date'].max()}")

    if "advisory_level" in df.columns:
        print("Advisory counts:", df["advisory_level"].value_counts().to_dict())

    if "predicted_fire_next_day" in df.columns:
        predicted_count = int(pd.to_numeric(df["predicted_fire_next_day"], errors="coerce").fillna(0).sum())
        print(f"Predicted next-day active rows: {predicted_count}")


if __name__ == "__main__":
    create_dashboard_dataset()