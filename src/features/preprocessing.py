import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import CONFIG
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

ID_COLUMN = CONFIG.data.id_column
TARGET_COLUMN = CONFIG.data.target_column
NUMERIC_FEATURES = list(CONFIG.features.numeric_features)
CATEGORICAL_FEATURES = list(CONFIG.features.categorical_features)
ENGINEERED_NUMERIC_FEATURES = list(CONFIG.features.engineered_numeric_features)
ENGINEERED_CATEGORICAL_FEATURES = list(CONFIG.features.engineered_categorical_features)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create leakage-safe features from existing employee attributes."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()

        # IDs and targets are never model inputs.
        data = data.drop(columns=[ID_COLUMN, TARGET_COLUMN], errors="ignore")

        data["salary_per_age"] = data["salary"] / data["age"].replace(0, np.nan)
        data["tenure_to_age_ratio"] = data["tenure_years"] / data["age"].replace(
            0,
            np.nan,
        )

        data["age_group"] = pd.cut(
            data["age"],
            bins=[0, 29, 39, 49, 59, np.inf],
            labels=["under_30", "30s", "40s", "50s", "60_plus"],
            include_lowest=True,
        ).astype("object")

        data["salary_band"] = pd.cut(
            data["salary"],
            bins=[-np.inf, 50000, 65000, 80000, np.inf],
            labels=["low_salary", "mid_salary", "high_salary", "very_high_salary"],
        ).astype("object")

        data["tenure_band"] = pd.cut(
            data["tenure_years"],
            bins=[-np.inf, 1, 3, 7, np.inf],
            labels=["new_hire", "early_tenure", "established", "long_tenure"],
        ).astype("object")

        data["dependents_employment_type"] = (
            data["has_dependents"].astype(str)
            + "_"
            + data["employment_type"].astype(str)
        )
        data["dependents_salary_band"] = (
            data["has_dependents"].astype(str) + "_" + data["salary_band"].astype(str)
        )
        data["age_dependents_group"] = (
            data["age_group"].astype(str) + "_" + data["has_dependents"].astype(str)
        )

        return data


def build_preprocessor() -> Pipeline:
    """Build the feature engineering and preprocessing pipeline.

    EDA showed the strongest model signals are dependents, employment type,
    salary, and age. The engineered features preserve those signals and add
    row-level interactions without using the target.
    """

    set_seed(CONFIG.data.random_state)

    numeric_features = NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
    categorical_features = CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES

    logger.info(
        "Building preprocessing pipeline with %d numeric and %d categorical features.",
        len(numeric_features),
        len(categorical_features),
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    column_preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("column_preprocessing", column_preprocessor),
        ]
    )
