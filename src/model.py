"""src/model.py"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from utils import get_logger


TARGET_COLUMN = "fire_next_day"
PREDICTION_COLUMN = "predicted_fire_next_day"
PROBABILITY_COLUMN = "predicted_fire_next_day_probability"
MODEL_FEATURES = [
    "fire_count",
    "avg_brightness",
    "max_brightness",
    "avg_confidence_score",
    "wind_speed",
    "alert_flag",
    "prev_day_fire_count",
    "prev_day_avg_brightness",
    "rolling_2day_fire_count",
    "rolling_3day_fire_count",
]
MIN_UNIQUE_DATES_FOR_MODEL = 7
URGENT_REVIEW_PROBABILITY = 0.7
PREPARE_PROBABILITY = 0.45
HIGH_ADVISORY_SCORE = 70
PREPARE_ADVISORY_SCORE = 45


def _empty_metrics() -> dict[str, Any]:
    """Return a starter metrics dictionary for graceful fallbacks."""
    return {
        "target": TARGET_COLUMN,
        "model_type": "RandomForestClassifier",
        "evaluation_status": "not_evaluated",
        "notes": [],
    }


def _safe_binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    """Compute readable binary classification metrics with zero-safe behavior."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select modeling columns and fill missing numeric values safely."""
    model_df = df.copy()
    for column in MODEL_FEATURES:
        if column not in model_df.columns:
            model_df[column] = 0

    model_df[MODEL_FEATURES] = model_df[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    return model_df


def _create_advisory_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a soft decision-support layer for situational awareness.

    These outputs are intentionally framed as review aids, not official emergency
    directives. A real operational system would need richer incident, perimeter,
    terrain, fuels, road, and agency data.
    """
    advisory_df = df.copy()
    if advisory_df.empty:
        advisory_df["advisory_score"] = pd.Series(dtype="int64")
        advisory_df["advisory_level"] = pd.Series(dtype="object")
        advisory_df["review_flag"] = pd.Series(dtype="int64")
        return advisory_df

    if "risk_level" not in advisory_df.columns:
        advisory_df["risk_level"] = "Low"
    if PREDICTION_COLUMN not in advisory_df.columns:
        advisory_df[PREDICTION_COLUMN] = 0
    if PROBABILITY_COLUMN not in advisory_df.columns:
        advisory_df[PROBABILITY_COLUMN] = 0.0

    risk_score_map = {"Low": 20, "Medium": 50, "High": 80}
    advisory_df["advisory_score"] = advisory_df["risk_level"].map(risk_score_map).fillna(20)
    advisory_df["advisory_score"] += advisory_df[PREDICTION_COLUMN].astype(int) * 15
    advisory_df["advisory_score"] += (advisory_df[PROBABILITY_COLUMN] >= PREPARE_PROBABILITY).astype(int) * 10
    advisory_df["advisory_score"] += (advisory_df[PROBABILITY_COLUMN] >= URGENT_REVIEW_PROBABILITY).astype(int) * 10
    advisory_df["advisory_score"] += (advisory_df["fire_count"] >= 3).astype(int) * 10
    advisory_df["advisory_score"] += (advisory_df["wind_speed"] >= 20).astype(int) * 10
    advisory_df["advisory_score"] += advisory_df["alert_flag"].astype(int) * 10
    advisory_df["advisory_score"] = advisory_df["advisory_score"].clip(lower=0, upper=100).round().astype(int)

    def assign_advisory_level(row: pd.Series) -> str:
        if (
            row["risk_level"] == "High"
            and (
                row[PROBABILITY_COLUMN] >= URGENT_REVIEW_PROBABILITY
                or row["alert_flag"] == 1
                or row["advisory_score"] >= HIGH_ADVISORY_SCORE
            )
        ):
            return "urgent_review"

        if (
            row["risk_level"] in {"Medium", "High"}
            or row[PREDICTION_COLUMN] == 1
            or row[PROBABILITY_COLUMN] >= PREPARE_PROBABILITY
            or row["advisory_score"] >= PREPARE_ADVISORY_SCORE
        ):
            return "prepare"

        return "monitor"

    advisory_df["advisory_level"] = advisory_df.apply(assign_advisory_level, axis=1)

    # review_flag marks bins that deserve closer human review for situational awareness.
    advisory_df["review_flag"] = advisory_df["advisory_level"].isin(["prepare", "urgent_review"]).astype(int)
    return advisory_df


def train_model(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Train a next-day wildfire activity model.

    The previous version predicted a hand-written risk label built from the same
    input features, which mostly taught the model to imitate rules. This version
    predicts whether the same spatial bin has fire activity on the next day.
    """
    logger = get_logger()
    logger.info("Training next-day wildfire forecasting model...")

    metrics = _empty_metrics()
    model_df = _prepare_model_frame(df)
    if model_df.empty:
        model_df[PREDICTION_COLUMN] = pd.Series(dtype="int64")
        model_df[PROBABILITY_COLUMN] = pd.Series(dtype="float64")
        model_df = _create_advisory_outputs(model_df)
        metrics["notes"].append("Input dataframe was empty.")
        return model_df, metrics

    if TARGET_COLUMN not in model_df.columns:
        logger.warning("Target column fire_next_day is missing from the feature set.")
        model_df[PREDICTION_COLUMN] = 0
        model_df[PROBABILITY_COLUMN] = 0.0
        model_df = _create_advisory_outputs(model_df)
        metrics["notes"].append("Target column fire_next_day was missing.")
        return model_df, metrics

    training_df = model_df.dropna(subset=["acq_date", TARGET_COLUMN]).copy()
    training_df["acq_date"] = pd.to_datetime(training_df["acq_date"], errors="coerce")
    training_df = training_df.dropna(subset=["acq_date"]).sort_values("acq_date").reset_index(drop=True)

    if training_df.empty:
        logger.warning("No rows were available for model training.")
        model_df[PREDICTION_COLUMN] = 0
        model_df[PROBABILITY_COLUMN] = 0.0
        model_df = _create_advisory_outputs(model_df)
        metrics["notes"].append("No rows were available after filtering training data.")
        return model_df, metrics

    target_distribution = training_df[TARGET_COLUMN].value_counts().to_dict()
    unique_dates = list(pd.Series(training_df["acq_date"].dt.normalize().dropna().unique()).sort_values())
    metrics["row_count"] = int(len(training_df))
    metrics["unique_date_count"] = int(len(unique_dates))
    metrics["class_distribution"] = {str(key): int(value) for key, value in target_distribution.items()}

    logger.info(f"Training rows: {len(training_df)}")
    logger.info(f"Unique forecast dates: {len(unique_dates)}")
    logger.info(f"Target distribution: {metrics['class_distribution']}")

    if len(unique_dates) < MIN_UNIQUE_DATES_FOR_MODEL:
        logger.warning(
            f"Only {len(unique_dates)} unique forecast dates are available. "
            f"Need at least {MIN_UNIQUE_DATES_FOR_MODEL} for meaningful forecasting. "
            "Skipping model training and using default prediction outputs."
        )
        model_df[PREDICTION_COLUMN] = 0
        model_df[PROBABILITY_COLUMN] = 0.0
        model_df = _create_advisory_outputs(model_df)
        metrics["evaluation_status"] = "skipped_insufficient_history"
        metrics["notes"].append(
            f"Only {len(unique_dates)} unique dates were available; need at least "
            f"{MIN_UNIQUE_DATES_FOR_MODEL} for meaningful time-based forecasting."
        )
        return model_df, metrics

    if training_df[TARGET_COLUMN].nunique() < 2:
        logger.warning("Only one target class is available. Falling back to a constant prediction.")
        constant_value = int(training_df[TARGET_COLUMN].iloc[0])
        model_df[PREDICTION_COLUMN] = constant_value
        model_df[PROBABILITY_COLUMN] = float(constant_value)
        model_df = _create_advisory_outputs(model_df)
        metrics["evaluation_status"] = "constant_prediction"
        metrics["constant_prediction"] = constant_value
        metrics["notes"].append("Only one class was present in the target.")
        return model_df, metrics

    test_date_count = max(1, int(round(len(unique_dates) * 0.25)))
    if len(unique_dates) - test_date_count < 1:
        test_date_count = 1

    train_dates = unique_dates[:-test_date_count]
    test_dates = unique_dates[-test_date_count:]

    if len(train_dates) == 0:
        logger.warning("Not enough dates for a clean train/test split. Using all rows for fitting only.")
        X_all = training_df[MODEL_FEATURES]
        y_all = training_df[TARGET_COLUMN].astype(int)
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        )
        model.fit(X_all, y_all)

        all_probabilities = model.predict_proba(model_df[MODEL_FEATURES])[:, 1]
        model_df[PROBABILITY_COLUMN] = all_probabilities
        model_df[PREDICTION_COLUMN] = (all_probabilities >= 0.5).astype(int)
        model_df = _create_advisory_outputs(model_df)
        metrics["evaluation_status"] = "fit_only"
        metrics["notes"].append("Dataset was too small for a proper time-aware split.")
        metrics["feature_importance"] = {
            feature: float(importance) for feature, importance in zip(MODEL_FEATURES, model.feature_importances_)
        }
        return model_df, metrics

    train_df = training_df[training_df["acq_date"].dt.normalize().isin(train_dates)].copy()
    test_df = training_df[training_df["acq_date"].dt.normalize().isin(test_dates)].copy()

    logger.info(
        "Using a time-aware split: train on earlier dates and test on later dates "
        f"({len(train_dates)} train dates, {len(test_dates)} test dates)."
    )

    if train_df.empty or test_df.empty:
        logger.warning("Time-aware split produced an empty train or test set. Falling back to fit-only mode.")
        X_all = training_df[MODEL_FEATURES]
        y_all = training_df[TARGET_COLUMN].astype(int)
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        )
        model.fit(X_all, y_all)

        all_probabilities = model.predict_proba(model_df[MODEL_FEATURES])[:, 1]
        model_df[PROBABILITY_COLUMN] = all_probabilities
        model_df[PREDICTION_COLUMN] = (all_probabilities >= 0.5).astype(int)
        model_df = _create_advisory_outputs(model_df)
        metrics["evaluation_status"] = "fit_only"
        metrics["notes"].append("Time-aware split was not possible after filtering.")
        metrics["feature_importance"] = {
            feature: float(importance) for feature, importance in zip(MODEL_FEATURES, model.feature_importances_)
        }
        return model_df, metrics

    X_train = train_df[MODEL_FEATURES]
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df[TARGET_COLUMN].astype(int)

    if y_train.nunique() < 2:
        logger.warning("Training split has only one class. Falling back to a constant prediction baseline.")
        constant_value = int(y_train.iloc[0])
        model_df[PREDICTION_COLUMN] = constant_value
        model_df[PROBABILITY_COLUMN] = float(constant_value)
        model_df = _create_advisory_outputs(model_df)
        metrics["evaluation_status"] = "constant_prediction"
        metrics["constant_prediction"] = constant_value
        metrics["notes"].append("Training split had only one class.")

        baseline_predictions = pd.Series([constant_value] * len(y_test), index=y_test.index)
        metrics.update(_safe_binary_metrics(y_test, baseline_predictions))
        return model_df, metrics

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)
    metrics.update(_safe_binary_metrics(y_test, test_predictions))
    metrics["evaluation_status"] = "time_split_evaluated"
    metrics["train_row_count"] = int(len(train_df))
    metrics["test_row_count"] = int(len(test_df))
    metrics["train_end_date"] = str(pd.Timestamp(max(train_dates)).date())
    metrics["test_start_date"] = str(pd.Timestamp(min(test_dates)).date())
    metrics["feature_importance"] = {
        feature: float(importance) for feature, importance in zip(MODEL_FEATURES, model.feature_importances_)
    }

    logger.info(
        "Evaluation metrics - "
        f"accuracy: {metrics['accuracy']:.3f}, "
        f"precision: {metrics['precision']:.3f}, "
        f"recall: {metrics['recall']:.3f}, "
        f"f1: {metrics['f1']:.3f}"
    )

    all_probabilities = model.predict_proba(model_df[MODEL_FEATURES])[:, 1]
    model_df[PROBABILITY_COLUMN] = all_probabilities
    model_df[PREDICTION_COLUMN] = (all_probabilities >= 0.5).astype(int)
    model_df = _create_advisory_outputs(model_df)
    return model_df, metrics
