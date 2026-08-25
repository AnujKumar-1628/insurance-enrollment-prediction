# Insurance Enrollment Prediction - Project Report

## Executive Summary

This project predicts whether an employee is likely to enroll in voluntary insurance based on demographic, employment, salary, dependent, and tenure information. The repository includes data validation, reproducible train/validation/test splitting, leakage-safe preprocessing, feature engineering, model training, MLflow experiment tracking, saved model artifacts, test-set evaluation, a FastAPI prediction service, and automated tests.

The final selected model is an XGBoost binary classifier because it achieved the best validation F1 score and is well suited for tabular data with non-linear relationships. A logistic regression model is also included as a transparent benchmark.

The saved final model achieved the following test performance:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.9990 |
| Precision | 1.0000 |
| Recall | 0.9984 |
| F1 | 0.9992 |
| ROC-AUC | 1.0000 |

These results are very high, so I also performed a validity check. The evidence suggests the dataset is almost rule-separable, meaning the high score is likely driven by the synthetic or deterministic structure of the data rather than proof that the model would automatically generalize to real-world insurance enrollment behavior.

## Problem Statement

The goal is to help predict employee enrollment in voluntary insurance. From a business perspective, this type of model could support benefits teams by identifying employees who may be more likely to enroll, understanding the factors associated with enrollment, and improving targeting for communication or outreach.

The prediction target is:

| Target | Meaning |
| --- | --- |
| `0` | Employee did not enroll |
| `1` | Employee enrolled |

The project is framed as a supervised binary classification problem.

## What I Built

The repository is organized as a small production-style ML project:

| Area | Purpose |
| --- | --- |
| `data/raw/employee_data.csv` | Original dataset |
| `data/processed/` | Train, validation, and test splits |
| `notebooks/01_eda.ipynb` | Exploratory data analysis |
| `notebooks/02_model_experiment.ipynb` | Wider model experimentation |
| `src/data/` | Data loading, validation, and splitting logic |
| `src/features/` | Feature engineering and preprocessing pipeline |
| `src/models/` | Model training and prediction utilities |
| `src/evaluation/` | Classification metric calculation |
| `scripts/` | Runnable scripts for data, training, evaluation, API, and tracking |
| `api/` | FastAPI service for single and batch predictions |
| `tests/` | Pytest coverage for data, preprocessing, models, and API |
| `models/` | Saved model artifacts |
| `reports/figures/` | Generated EDA and model evaluation figures |
| `mlflow.db`, `mlruns/` | Local MLflow experiment tracking |

This structure shows the complete process: data quality checks, model development, evaluation, tracking, packaging, serving, and testing.

## My Process

### 1. Data Understanding

I first inspected the dataset to understand the columns, target distribution, missing values, and relationships between features and enrollment. The dataset contains 10,000 employee records and 10 columns:

- `employee_id`
- `age`
- `gender`
- `marital_status`
- `salary`
- `employment_type`
- `region`
- `has_dependents`
- `tenure_years`
- `enrolled`

There were no missing values in the raw dataset. The target was moderately imbalanced:

| Class | Meaning | Count | Share |
| --- | --- | ---: | ---: |
| 0 | Not enrolled | 3,826 | 38.26% |
| 1 | Enrolled | 6,174 | 61.74% |

Important observations from EDA:

- Employees with dependents had a much higher enrollment rate.
- Full-time employees enrolled at a higher rate than contract or part-time employees.
- Enrolled employees were generally older and had higher salaries.
- Tenure showed weaker separation compared with salary, age, employment type, and dependent status.

### 2. Data Validation and Splitting

I added validation before training so data issues are caught early. The validation checks include:

- Required columns are present.
- The dataset is not empty.
- `employee_id` values are unique.
- Required fields do not contain missing values.
- The target only contains `0` and `1`.
- Age, salary, and tenure values are within valid ranges.

After validation, I split the data into train, validation, and test sets:

| Split | Rows | Purpose |
| --- | ---: | --- |
| Train | 6,000 | Fit preprocessing and model parameters |
| Validation | 2,000 | Compare models and select the final candidate |
| Test | 2,000 | Final unbiased evaluation of the saved model |

The split is stratified by the target so the class balance is preserved across train, validation, and test sets.

### 3. Leakage-Safe Preprocessing

I designed preprocessing as an sklearn `Pipeline` so transformations are fitted only on the training data. This prevents train/test leakage and makes the exact same transformations available during API inference.

The preprocessing pipeline includes:

- Dropping `employee_id` and `enrolled` from model inputs.
- Median imputation for numeric features.
- Standard scaling for numeric features.
- Most-frequent imputation for categorical features.
- One-hot encoding for categorical features with unknown-category handling.
- Feature engineering inside the pipeline.

Feature engineering includes:

| Feature | Why it was added |
| --- | --- |
| `salary_per_age` | Captures salary level relative to age |
| `tenure_to_age_ratio` | Captures tenure relative to career stage |
| `age_group` | Adds non-linear age buckets |
| `salary_band` | Adds salary groups that tree and linear models can use |
| `tenure_band` | Adds interpretable tenure groups |
| `dependents_employment_type` | Captures interaction between family status and employment type |
| `dependents_salary_band` | Captures interaction between dependent status and salary level |
| `age_dependents_group` | Captures interaction between age group and dependent status |

I kept the feature engineering based only on input fields, not the target, so it remains safe for validation, testing, and production inference.

### 4. Model Experimentation

In the experimentation notebook, I compared several model families:

| Model | Reason for Testing |
| --- | --- |
| Logistic Regression | Simple, fast, explainable linear baseline |
| Random Forest | Non-linear tree model with bagging |
| Gradient Boosting | Boosted tree model for tabular patterns |
| SGD Logistic Regression | Faster stochastic linear baseline |
| XGBoost | Strong regularized boosted model for tabular data |

After experimentation, I narrowed the production training script to two models:

- Logistic regression as the benchmark model.
- XGBoost as the final candidate model.

This keeps the final pipeline focused while still showing that I explored alternatives before selecting the production candidate.

### 5. Why I Chose These Models

Logistic regression was included because it is interpretable, reliable, and gives a strong benchmark. It also works well with standardized numeric features and one-hot encoded categorical features. I used balanced class weights to account for the target imbalance.

XGBoost was selected as the final model because the EDA suggested interactions between dependents, employment type, salary, and age. A boosted tree model can capture these relationships more naturally than a purely linear model. I also used regularization-related parameters such as shallow tree depth, `min_child_weight`, subsampling, column sampling, `reg_alpha`, and `reg_lambda` to reduce overfitting risk.

## Evaluation Results

### Notebook Validation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9645 | 0.9842 | 0.9579 | 0.9709 | 0.9939 |
| Regularized Random Forest | 0.9955 | 0.9936 | 0.9992 | 0.9964 | 1.0000 |
| Regularized Gradient Boosting | 0.9995 | 1.0000 | 0.9992 | 0.9996 | 1.0000 |
| SGD Logistic Regression | 0.9460 | 0.9804 | 0.9312 | 0.9551 | 0.9900 |
| Regularized XGBoost | 0.9985 | 1.0000 | 0.9976 | 0.9988 | 1.0000 |

### Production Training Script Validation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9645 | 0.9842 | 0.9579 | 0.9709 | 0.9939 |
| XGBoost | 0.9985 | 1.0000 | 0.9976 | 0.9988 | 1.0000 |

The best production candidate is XGBoost by validation F1 score.

### Final Test Results

The saved final model was evaluated on the held-out test split:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.9990 |
| Precision | 1.0000 |
| Recall | 0.9984 |
| F1 | 0.9992 |
| ROC-AUC | 1.0000 |

Test confusion matrix:

|  | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| Actual 0 | 765 | 0 |
| Actual 1 | 2 | 1,233 |

The model made only two false negative predictions and zero false positive predictions on the test set.

## High-Score Validity Check

Because the scores are near perfect, I did not accept them blindly. I checked whether there was obvious leakage in the implementation:

- Rows are split before preprocessing.
- The target column is removed from model inputs.
- `employee_id` is removed from model inputs.
- Imputers, scalers, encoders, and engineered features are inside sklearn pipelines.
- Class weighting is calculated using the training target only.

I did not find obvious classic leakage in the project code.

However, additional diagnostics showed the raw dataset itself appears almost perfectly separable. For example:

| Dependents | Employment type | Enrollment rate |
| --- | --- | ---: |
| No | Contract | 0.00% |
| No | Part-time | 0.00% |
| Yes | Full-time | 92.78% |

A shallow diagnostic decision tree trained only on the original non-ID features reached 0.9995 test accuracy. A simple rule derived from that tree matched 9,998 of the 10,000 labels.

My interpretation is that the model is probably learning a simple underlying rule in the dataset. This is useful for the assessment because it shows the pipeline works, but in a real business setting I would not claim production readiness until testing on independent data from the real enrollment process.

## MLflow Tracking

I added local MLflow tracking so model runs can be compared and reproduced. The project stores:

- Model parameters.
- Validation metrics.
- Model artifacts.
- Run metadata.

The tracking files are stored locally:

| File/Folder | Purpose |
| --- | --- |
| `mlflow.db` | SQLite backend for MLflow metadata |
| `mlruns/` | MLflow artifacts and saved model files |

This makes the project easier to audit because a reviewer can see what was trained, with which parameters, and how it performed.

## API Serving

I built a FastAPI service around the saved model so the project demonstrates how the model could be used by another system.

Available endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Checks API status and whether the model is loaded |
| `/predict` | POST | Returns one employee enrollment prediction |
| `/predict/batch` | POST | Returns predictions for multiple employee records |

Example request:

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

Example response:

```json
{
  "enrollment_probability": 0.76,
  "enrolled_prediction": 1
}
```

The API loads the trained pipeline once at startup, so prediction requests use the same preprocessing and model logic that was used during training.

## Testing

I added automated tests to protect the main project behavior:

| Test Area | What It Checks |
| --- | --- |
| Data tests | Raw data validation and split behavior |
| Preprocessing tests | Feature engineering and pipeline transformation |
| Model tests | Training, prediction, and model output behavior |
| API tests | Health check, single prediction, batch prediction, and invalid inputs |

The API tests use a fake model where appropriate, which keeps the tests fast and focused on API behavior rather than retraining a model.


## Key Decisions and Why They Matter

| Decision | Why I Made It |
| --- | --- |
| Used train/validation/test splits | Keeps model selection and final evaluation separate |
| Used stratified splitting | Preserves target distribution across splits |
| Put preprocessing inside sklearn pipelines | Prevents leakage and keeps training/inference consistent |
| Removed `employee_id` from features | Avoids learning from an identifier |
| Kept logistic regression benchmark | Provides a simple explainable comparison point |
| Selected XGBoost final model | Best validation F1 and strong fit for tabular interactions |
| Added MLflow | Makes experiments trackable and reproducible |
| Added FastAPI | Shows how the model can be served to applications |
| Added pytest tests | Protects data, model, preprocessing, and API behavior |

## Limitations

The main limitation is the dataset structure. The model performs almost perfectly because the labels appear to follow a very strong rule. This means the project is technically correct, but the score should be interpreted carefully.

Other limitations:

- The dataset may not represent real-world employee behavior.
- The model was not validated on a later time period or external dataset.
- Fairness and bias analysis by demographic groups would be needed before real use.
- Probability calibration should be checked before using the score as a business risk or targeting score.
- The decision threshold is currently `0.50`; in production, it should be tuned based on business costs.


## Final Takeaway

This project demonstrates an end-to-end machine learning workflow for insurance enrollment prediction. I handled the work from data validation through model training, evaluation, tracking, API serving, and tests. XGBoost is the best-performing model in this dataset, but I also documented why the near-perfect score should be treated with caution. That combination of strong implementation and careful interpretation is the main value of the project.
