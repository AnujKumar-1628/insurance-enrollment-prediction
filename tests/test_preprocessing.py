from src.features.preprocessing import FeatureEngineer, build_preprocessor


def test_feature_engineer_adds_expected_features(employee_rows):
    features = employee_rows.drop(columns=["employee_id", "enrolled"])

    transformed = FeatureEngineer().fit_transform(features)

    assert "salary_per_age" in transformed.columns
    assert "tenure_to_age_ratio" in transformed.columns
    assert "salary_band" in transformed.columns
    assert "dependents_employment_type" in transformed.columns
    assert transformed.loc[0, "salary_per_age"] == 52000.0 / 25


def test_feature_engineer_drops_id_and_target_if_present(employee_rows):
    transformed = FeatureEngineer().fit_transform(employee_rows)

    assert "employee_id" not in transformed.columns
    assert "enrolled" not in transformed.columns


def test_preprocessor_fits_and_transforms_employee_features(employee_rows):
    X = employee_rows.drop(columns=["employee_id", "enrolled"])

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > X.shape[1]
