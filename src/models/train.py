from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.evaluation.evaluate import classification_metrics
from src.features.preprocessing import build_preprocessor
from src.utils.config import CONFIG
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.tracking import log_model_run, tracking_params


logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    model: Pipeline
    validation_metrics: dict[str, float]
    artifact_path: Path


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from configured target and identifier columns."""

    drop_columns = [CONFIG.data.target_column, CONFIG.data.id_column]
    X = data.drop(columns=drop_columns)
    y = data[CONFIG.data.target_column]
    return X, y


def load_training_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load processed train and validation splits."""

    train = pd.read_csv(CONFIG.data.train_data_path)
    validation = pd.read_csv(CONFIG.data.validation_data_path)

    X_train, y_train = split_features_target(train)
    X_validation, y_validation = split_features_target(validation)

    logger.info("Loaded train split: %d rows", len(train))
    logger.info("Loaded validation split: %d rows", len(validation))

    return X_train, y_train, X_validation, y_validation


def build_pipeline(classifier) -> Pipeline:
    """Attach a fresh preprocessing pipeline to a classifier."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def build_benchmark_model() -> Pipeline:
    """Build logistic regression benchmark model."""

    classifier = LogisticRegression(
        max_iter=CONFIG.model.logistic_max_iter,
        class_weight="balanced",
        random_state=CONFIG.data.random_state,
    )
    return build_pipeline(classifier)


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """Calculate XGBoost class weighting from the training target only."""

    negative_count = (y == 0).sum()
    positive_count = (y == 1).sum()
    return negative_count / positive_count


def build_xgboost_model(y_train: pd.Series) -> Pipeline:
    """Build configured XGBoost model."""

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=CONFIG.model.xgboost_n_estimators,
        learning_rate=CONFIG.model.xgboost_learning_rate,
        max_depth=CONFIG.model.xgboost_max_depth,
        min_child_weight=CONFIG.model.xgboost_min_child_weight,
        subsample=CONFIG.model.subsample,
        colsample_bytree=CONFIG.model.colsample_bytree,
        reg_alpha=CONFIG.model.reg_alpha,
        reg_lambda=CONFIG.model.reg_lambda,
        scale_pos_weight=calculate_scale_pos_weight(y_train),
        random_state=CONFIG.data.random_state,
        n_jobs=1,
    )
    return build_pipeline(classifier)


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Evaluate a fitted classification pipeline."""

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return classification_metrics(y, y_pred, y_proba)


def save_model(model: Pipeline, path: Path) -> None:
    """Persist a fitted model pipeline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def train_single_model(
    model_name: str,
    model: Pipeline,
    artifact_path: Path,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> TrainingResult:
    """Train, validate, and save one model."""

    logger.info("Training %s", model_name)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_validation, y_validation)
    logger.info("%s validation metrics: %s", model_name, metrics)

    save_model(model, artifact_path)
    log_model_run(
        run_name=model_name,
        params=tracking_params(model_name),
        metrics=metrics,
        artifact_path=artifact_path,
    )

    return TrainingResult(
        model_name=model_name,
        model=model,
        validation_metrics=metrics,
        artifact_path=artifact_path,
    )


def train_models() -> dict[str, TrainingResult]:
    """Train logistic regression benchmark and XGBoost final candidate."""

    set_seed(CONFIG.data.random_state)
    X_train, y_train, X_validation, y_validation = load_training_data()

    benchmark = train_single_model(
        model_name=CONFIG.model.benchmark_model_type,
        model=build_benchmark_model(),
        artifact_path=CONFIG.model.benchmark_model_path,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    final_candidate = train_single_model(
        model_name=CONFIG.model.final_model_type,
        model=build_xgboost_model(y_train),
        artifact_path=CONFIG.model.final_model_path,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    return {
        benchmark.model_name: benchmark,
        final_candidate.model_name: final_candidate,
    }


def get_best_model(results: dict[str, TrainingResult]) -> TrainingResult:
    """Select the best model by validation F1 score."""

    return max(
        results.values(),
        key=lambda result: result.validation_metrics["f1"],
    )
