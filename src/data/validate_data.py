import pandas as pd


REQUIRED_COLUMNS = {
    "employee_id",
    "age",
    "gender",
    "marital_status",
    "salary",
    "employment_type",
    "region",
    "has_dependents",
    "tenure_years",
    "enrolled",
}


TARGET_VALUES = {0, 1}


def validate_employee_data(df: pd.DataFrame) -> None:
    """Validate the raw employee enrollment dataset before splitting.

    These checks run before model training so data issues are caught early.
    Feature engineering and imputation should happen later inside the model
    pipeline to avoid train/test leakage.
    """

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if df["employee_id"].duplicated().any():
        raise ValueError("employee_id contains duplicate values.")

    missing_counts = df[list(REQUIRED_COLUMNS)].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(f"Dataset contains missing values: {missing_counts.to_dict()}")

    target_values = set(df["enrolled"].unique())
    invalid_targets = target_values.difference(TARGET_VALUES)
    if invalid_targets:
        raise ValueError(f"Invalid enrolled values found: {sorted(invalid_targets)}")

    if (df["age"] < 16).any() or (df["age"] > 100).any():
        raise ValueError("age contains values outside the expected range 16-100.")

    if (df["salary"] < 0).any():
        raise ValueError("salary contains negative values.")

    if (df["tenure_years"] < 0).any():
        raise ValueError("tenure_years contains negative values.")
