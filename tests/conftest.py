import pandas as pd
import pytest


@pytest.fixture
def employee_rows() -> pd.DataFrame:
    """Small valid employee dataset with both target classes."""

    return pd.DataFrame(
        [
            {
                "employee_id": 1,
                "age": 25,
                "gender": "Female",
                "marital_status": "Single",
                "salary": 52000.0,
                "employment_type": "Full-time",
                "region": "North",
                "has_dependents": False,
                "tenure_years": 1.5,
                "enrolled": 0,
            },
            {
                "employee_id": 2,
                "age": 42,
                "gender": "Male",
                "marital_status": "Married",
                "salary": 86000.0,
                "employment_type": "Full-time",
                "region": "South",
                "has_dependents": True,
                "tenure_years": 8.0,
                "enrolled": 1,
            },
            {
                "employee_id": 3,
                "age": 34,
                "gender": "Female",
                "marital_status": "Married",
                "salary": 69000.0,
                "employment_type": "Part-time",
                "region": "East",
                "has_dependents": True,
                "tenure_years": 4.0,
                "enrolled": 1,
            },
            {
                "employee_id": 4,
                "age": 58,
                "gender": "Male",
                "marital_status": "Divorced",
                "salary": 47000.0,
                "employment_type": "Contract",
                "region": "West",
                "has_dependents": False,
                "tenure_years": 0.5,
                "enrolled": 0,
            },
            {
                "employee_id": 5,
                "age": 29,
                "gender": "Female",
                "marital_status": "Single",
                "salary": 61000.0,
                "employment_type": "Full-time",
                "region": "North",
                "has_dependents": False,
                "tenure_years": 3.0,
                "enrolled": 0,
            },
            {
                "employee_id": 6,
                "age": 47,
                "gender": "Male",
                "marital_status": "Married",
                "salary": 93000.0,
                "employment_type": "Full-time",
                "region": "South",
                "has_dependents": True,
                "tenure_years": 12.0,
                "enrolled": 1,
            },
            {
                "employee_id": 7,
                "age": 39,
                "gender": "Female",
                "marital_status": "Married",
                "salary": 78000.0,
                "employment_type": "Part-time",
                "region": "East",
                "has_dependents": True,
                "tenure_years": 5.5,
                "enrolled": 1,
            },
            {
                "employee_id": 8,
                "age": 23,
                "gender": "Male",
                "marital_status": "Single",
                "salary": 43000.0,
                "employment_type": "Contract",
                "region": "West",
                "has_dependents": False,
                "tenure_years": 0.8,
                "enrolled": 0,
            },
            {
                "employee_id": 9,
                "age": 51,
                "gender": "Female",
                "marital_status": "Married",
                "salary": 101000.0,
                "employment_type": "Full-time",
                "region": "North",
                "has_dependents": True,
                "tenure_years": 15.0,
                "enrolled": 1,
            },
            {
                "employee_id": 10,
                "age": 31,
                "gender": "Male",
                "marital_status": "Single",
                "salary": 56000.0,
                "employment_type": "Part-time",
                "region": "South",
                "has_dependents": False,
                "tenure_years": 2.0,
                "enrolled": 0,
            },
        ]
    )


@pytest.fixture
def employee_record() -> dict:
    """Single API/model input record without id or target columns."""

    return {
        "age": 42,
        "gender": "Male",
        "marital_status": "Married",
        "salary": 86000.0,
        "employment_type": "Full-time",
        "region": "South",
        "has_dependents": True,
        "tenure_years": 8.0,
    }
