from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.utils.config import CONFIG
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_model(model_path: Path | None = None) -> Pipeline:
    """Load a trained model pipeline."""

    path = model_path or CONFIG.model.model_path
    logger.info("Loading model from %s", path)
    return joblib.load(path)


def records_to_frame(records: pd.DataFrame | dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    """Convert supported prediction inputs into a pandas DataFrame."""

    if isinstance(records, pd.DataFrame):
        return records.copy()

    if isinstance(records, dict):
        return pd.DataFrame([records])

    return pd.DataFrame(records)


def predict_proba(
    records: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    model: Pipeline | None = None,
) -> list[float]:
    """Return enrollment probabilities for one or more records."""

    fitted_model = model or load_model()
    features = records_to_frame(records)
    probabilities = fitted_model.predict_proba(features)[:, 1]
    return probabilities.tolist()


def predict_enrollment(
    records: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    model: Pipeline | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Return enrollment predictions and probabilities."""

    cutoff = CONFIG.model.threshold if threshold is None else threshold
    probabilities = predict_proba(records, model=model)
    predictions = [int(probability >= cutoff) for probability in probabilities]

    return pd.DataFrame(
        {
            "enrollment_probability": probabilities,
            "enrolled_prediction": predictions,
        }
    )
