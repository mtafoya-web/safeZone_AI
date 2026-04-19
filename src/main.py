"""src/main.py"""

from __future__ import annotations

import json

from clean import clean_data
from data_fetch import fetch_firms_data
from features import add_weather, create_labels, engineer_features
from model import train_model
from utils import OUTPUT_DIR, ensure_directories, get_logger, load_environment


def _log_stage_rows(stage_name: str, row_count: int) -> None:
    """Print a consistent row count message after each pipeline stage."""
    get_logger().info(f"Rows after {stage_name}: {row_count}")


def _stop_if_empty(df, stage_name: str) -> bool:
    """
    Stop the pipeline early when a stage returns no rows.

    This keeps the pipeline beginner-friendly and makes failures obvious.
    """
    if df.empty:
        get_logger().warning(
            f"Pipeline stopped after {stage_name} because the dataframe is empty. "
            "Check the debug logs above for the likely cause."
        )
        return True
    return False


def run_pipeline() -> None:
    """
    Run the wildfire pipeline from raw detections to next-day forecasting outputs.

    The project now tells two connected stories:
    1. a rule-based current-day risk label for explainability
    2. an ML-based next-day fire activity forecast at the spatial-bin level
    """
    logger = get_logger()
    logger.info("Starting wildfire forecasting pipeline...")

    try:
        load_environment()
        ensure_directories()

        raw_df = fetch_firms_data()
        _log_stage_rows("fetch", len(raw_df))
        if _stop_if_empty(raw_df, "fetch"):
            return

        cleaned_df = clean_data(raw_df)
        _log_stage_rows("clean", len(cleaned_df))
        if _stop_if_empty(cleaned_df, "clean"):
            return

        featured_df = engineer_features(cleaned_df)
        _log_stage_rows("features", len(featured_df))
        if _stop_if_empty(featured_df, "features"):
            return

        weather_df = add_weather(featured_df)
        _log_stage_rows("weather", len(weather_df))
        if _stop_if_empty(weather_df, "weather"):
            return

        labeled_df = create_labels(weather_df)
        _log_stage_rows("labels", len(labeled_df))
        if _stop_if_empty(labeled_df, "labels"):
            return

        predictions_df, metrics = train_model(labeled_df)
        _log_stage_rows("model", len(predictions_df))
        if _stop_if_empty(predictions_df, "model"):
            return

        if "risk_level" in predictions_df.columns:
            logger.info(f"High risk bins: {int((predictions_df['risk_level'] == 'High').sum())}")
        if "predicted_fire_next_day" in predictions_df.columns:
            logger.info(
                f"Predicted next-day active bins: {int((predictions_df['predicted_fire_next_day'] == 1).sum())}"
            )
        if "advisory_level" in predictions_df.columns:
            logger.info(f"Advisory level counts: {predictions_df['advisory_level'].value_counts().to_dict()}")
        if "review_flag" in predictions_df.columns:
            logger.info(f"Review flag counts: {predictions_df['review_flag'].value_counts().to_dict()}")

        output_path = OUTPUT_DIR / "predictions.csv"
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Saved final predictions to {output_path}")

        metrics_path = OUTPUT_DIR / "model_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)
        logger.info(f"Saved model metrics to {metrics_path}")

        logger.info("Pipeline complete.")
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}")


if __name__ == "__main__":
    run_pipeline()
