import numpy as np
import pandas as pd

from src.models.predict import predict_enrollment, predict_proba, records_to_frame


class FakeProbabilityModel:
    def __init__(self, positive_probabilities):
        self.positive_probabilities = positive_probabilities

    def predict_proba(self, features):
        negative_probabilities = [
            1 - probability for probability in self.positive_probabilities
        ]
        return np.column_stack([negative_probabilities, self.positive_probabilities])


def test_records_to_frame_converts_single_record(employee_record):
    frame = records_to_frame(employee_record)

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 1
    assert frame.loc[0, "age"] == employee_record["age"]


def test_records_to_frame_copies_dataframe(employee_rows):
    original = employee_rows.drop(columns=["employee_id", "enrolled"])

    frame = records_to_frame(original)

    assert frame.equals(original)
    assert frame is not original


def test_predict_proba_returns_positive_class_probabilities(employee_record):
    model = FakeProbabilityModel([0.73])

    probabilities = predict_proba(employee_record, model=model)

    assert probabilities == [0.73]


def test_predict_enrollment_applies_threshold(employee_record):
    records = [employee_record, employee_record]
    model = FakeProbabilityModel([0.49, 0.51])

    predictions = predict_enrollment(records, model=model, threshold=0.5)

    assert predictions["enrollment_probability"].tolist() == [0.49, 0.51]
    assert predictions["enrolled_prediction"].tolist() == [0, 1]
