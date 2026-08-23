import pytest

from src.data.load_data import DataLoader
from src.data.validate_data import validate_employee_data


def test_validate_employee_data_accepts_valid_rows(employee_rows):
    validate_employee_data(employee_rows)


def test_validate_employee_data_rejects_missing_required_column(employee_rows):
    invalid_rows = employee_rows.drop(columns=["salary"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_employee_data(invalid_rows)


def test_validate_employee_data_rejects_duplicate_employee_id(employee_rows):
    invalid_rows = employee_rows.copy()
    invalid_rows.loc[1, "employee_id"] = invalid_rows.loc[0, "employee_id"]

    with pytest.raises(ValueError, match="duplicate"):
        validate_employee_data(invalid_rows)


def test_validate_employee_data_rejects_invalid_target(employee_rows):
    invalid_rows = employee_rows.copy()
    invalid_rows.loc[0, "enrolled"] = 2

    with pytest.raises(ValueError, match="Invalid enrolled values"):
        validate_employee_data(invalid_rows)


def test_split_data_removes_id_and_target_from_features(employee_rows):
    loader = DataLoader()
    loader.test_size = 0.2
    loader.validation_size = 0.2
    loader.random_state = 42

    splits = loader.split_data(employee_rows)

    assert len(splits.train) + len(splits.validation) + len(splits.test) == len(
        employee_rows
    )
    assert "employee_id" not in splits.X_train.columns
    assert "enrolled" not in splits.X_train.columns
    assert set(splits.y_train.unique()).issubset({0, 1})
