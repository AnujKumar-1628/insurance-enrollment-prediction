from dataclasses import dataclass, field
from pathlib import Path

from src.utils.path import (
    MLFLOW_DB_PATH,
    MLRUNS_DIR,
    MODEL_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_PATH,
)


@dataclass(frozen=True)
class DataConfig:
    raw_data_path: Path = RAW_DATA_PATH
    processed_data_dir: Path = PROCESSED_DATA_DIR
    train_data_path: Path = PROCESSED_DATA_DIR / "train.csv"
    validation_data_path: Path = PROCESSED_DATA_DIR / "validation.csv"
    test_data_path: Path = PROCESSED_DATA_DIR / "test.csv"

    target_column: str = "enrolled"
    id_column: str = "employee_id"
    test_size: float = 0.20
    validation_size: float = 0.20
    random_state: int = 42


@dataclass(frozen=True)
class FeatureConfig:
    numeric_features: tuple[str, ...] = ("age", "salary", "tenure_years")
    categorical_features: tuple[str, ...] = (
        "gender",
        "marital_status",
        "employment_type",
        "region",
        "has_dependents",
    )
    engineered_numeric_features: tuple[str, ...] = (
        "salary_per_age",
        "tenure_to_age_ratio",
    )
    engineered_categorical_features: tuple[str, ...] = (
        "age_group",
        "salary_band",
        "tenure_band",
        "dependents_employment_type",
        "dependents_salary_band",
        "age_dependents_group",
    )


@dataclass(frozen=True)
class ModelConfig:
    model_path: Path = MODEL_PATH
    benchmark_model_path: Path = MODELS_DIR / "benchmark_logistic_regression.joblib"
    final_model_path: Path = MODEL_PATH
    benchmark_model_type: str = "logistic_regression"
    final_model_type: str = "xgboost"
    threshold: float = 0.50
    logistic_max_iter: int = 2000
    xgboost_n_estimators: int = 120
    xgboost_learning_rate: float = 0.05
    xgboost_max_depth: int = 3
    xgboost_min_child_weight: int = 10
    subsample: float = 0.80
    colsample_bytree: float = 0.80
    reg_alpha: float = 0.10
    reg_lambda: float = 5.00


@dataclass(frozen=True)
class TrackingConfig:
    experiment_name: str = "insurance-enrollment-prediction"
    mlflow_tracking_uri: str = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
    wandb_project: str = "insurance-enrollment-prediction"
    wandb_mode: str = "offline"
    enable_mlflow: bool = True
    enable_wandb: bool = False


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


CONFIG = Config()
