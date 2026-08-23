# Insurance Enrollment Prediction

Machine learning project for predicting whether an employee will enroll in voluntary insurance.

The project includes:

- Data validation and train/validation/test splitting
- Leakage-safe preprocessing and feature engineering
- Logistic regression benchmark model
- XGBoost final candidate model
- MLflow experiment tracking
- FastAPI prediction service
- Pytest test suite

## Project Structure

```text
api/                    FastAPI app and request/response schemas
data/raw/               Raw input data
data/processed/         Train, validation, and test CSV files
models/                 Saved trained model artifacts
notebooks/              EDA and model experimentation notebooks
reports/figures/        Generated plots and evaluation figures
scripts/                Runnable project scripts
src/                    Data, feature, model, evaluation, and utility code
tests/                  Automated tests
report.md               Project summary report
requirements.txt        Python dependencies
```

## Setup

Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Data Pipeline

This validates the raw dataset and creates processed train, validation, and test splits.

```powershell
python scripts/run_data.py
```

Outputs:

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

## Train Models

Train the benchmark logistic regression model and the XGBoost final candidate.

```powershell
python scripts/train_model.py
```

Outputs:

```text
models/benchmark_logistic_regression.joblib
models/enrollment_model.joblib
```

## Run MLflow Tracking

Run experiment tracking around the existing project training code.

```powershell
python scripts/mlflow_experiment.py
```

Start the MLflow UI.

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Open:

```text
http://127.0.0.1:5000
```

Look for the experiment:

```text
insurance-enrollment-tracking
```

MLflow storage:

```text
mlflow.db = run metadata, params, metrics
mlruns/   = model artifacts and files
```


## Evaluate the Saved Model

Evaluate `models/enrollment_model.joblib` on the test split.

```powershell
python scripts/evaluate_model.py
```

Output figure:

```text
reports/figures/confusion_matrix_test.png
```

## Run the API

Start the FastAPI prediction service.

```powershell
python scripts/run_api.py
```

For development reload:

```powershell
python scripts/run_api.py --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Example `/predict` payload:

```json
{
  "age": 42,
  "gender": "Male",
  "marital_status": "Married",
  "salary": 86000,
  "employment_type": "Full-time",
  "region": "South",
  "has_dependents": true,
  "tenure_years": 8
}
```

## Run Tests

```powershell
python -m pytest tests -p no:cacheprovider
```

The `-p no:cacheprovider` flag avoids local pytest cache permission issues on some Windows setups.

## Current Results

Validation results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9645 | 0.9842 | 0.9579 | 0.9709 | 0.9939 |
| XGBoost | 0.9985 | 1.0000 | 0.9976 | 0.9988 | 1.0000 |

The current best model is XGBoost by validation F1 score.

See [report.md](report.md) for the project summary, observations, evaluation results, and next steps.
