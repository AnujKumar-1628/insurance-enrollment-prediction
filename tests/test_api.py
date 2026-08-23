import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main


class FakeModel:
    def predict_proba(self, features):
        positive_probabilities = np.full(len(features), 0.76)
        negative_probabilities = 1 - positive_probabilities
        return np.column_stack([negative_probabilities, positive_probabilities])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_main, "load_model", lambda model_path: FakeModel())
    api_main.model = None

    with TestClient(api_main.app) as test_client:
        yield test_client

    api_main.model = None


def test_health_check_returns_model_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_path"].replace("/", "\\").endswith(
        "models\\enrollment_model.joblib"
    )


def test_predict_returns_enrollment_prediction(client, employee_record):
    response = client.post("/predict", json=employee_record)

    assert response.status_code == 200
    assert response.json() == {
        "enrollment_probability": 0.76,
        "enrolled_prediction": 1,
    }


def test_predict_batch_returns_one_prediction_per_record(client, employee_record):
    response = client.post(
        "/predict/batch",
        json={"records": [employee_record, employee_record]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "predictions": [
            {"enrollment_probability": 0.76, "enrolled_prediction": 1},
            {"enrollment_probability": 0.76, "enrolled_prediction": 1},
        ]
    }


def test_predict_rejects_missing_required_field(client, employee_record):
    invalid_record = employee_record.copy()
    invalid_record.pop("salary")

    response = client.post("/predict", json=invalid_record)

    assert response.status_code == 422


def test_predict_rejects_invalid_numeric_values(client, employee_record):
    invalid_record = employee_record.copy()
    invalid_record["age"] = 12

    response = client.post("/predict", json=invalid_record)

    assert response.status_code == 422
