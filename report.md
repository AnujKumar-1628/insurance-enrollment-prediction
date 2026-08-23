# Insurance Enrollment Prediction Report

## Data Observations

The dataset contains 10,000 employee records with 10 columns. The target column is `enrolled`, and each row also includes an `employee_id`, demographic information, employment details, salary, dependent status, and tenure.

The processed split is:

| Split | Rows |
| --- | ---: |
| Train | 6,000 |
| Validation | 2,000 |
| Test | 2,000 |

There are no missing values in the raw dataset. The target is moderately imbalanced:

| Class | Meaning | Count | Share |
| --- | --- | ---: | ---: |
| 0 | Not enrolled | 3,826 | 38.26% |
| 1 | Enrolled | 6,174 | 61.74% |

Notable patterns:

- Employees with dependents have a much higher enrollment rate than employees without dependents.
- Full-time employees enroll at a higher rate than contract or part-time employees.
- Enrolled employees are older on average and have higher average salaries.
- Tenure is less separated between the two classes than salary, age, employment type, and dependent status.

## Model Choices and Rationale

The production training script compares two supervised binary classification models.

The benchmark model is logistic regression. It is a good baseline because it is simple, fast, interpretable, and works well with standardized numeric features plus one-hot encoded categorical features. The model uses balanced class weights to account for the target imbalance.

The final candidate model is XGBoost. It was selected because tree-based boosting models usually perform well on tabular business datasets with non-linear relationships and feature interactions. In this project, enrollment is likely influenced by interactions such as dependents plus employment type, salary bands, and age groups, which XGBoost can capture more naturally than a linear model.

`notebooks/02_model_experiment.ipynb` was used to compare a wider set of model types before narrowing the project down to the logistic regression benchmark and XGBoost final candidate:

- Logistic regression: linear, interpretable baseline.
- Random forest: bagged tree model for non-linear patterns.
- Gradient boosting: boosted tree model with shallow trees and shrinkage.
- SGD logistic regression: faster linear logistic variant trained with stochastic gradient descent.
- XGBoost: boosted tree model with stronger tabular-data performance and regularization.

Both models use the same preprocessing pipeline:

- Median imputation and scaling for numeric features
- Most-frequent imputation and one-hot encoding for categorical features
- Leakage-safe feature engineering inside the sklearn pipeline
- Engineered features such as `salary_per_age`, `tenure_to_age_ratio`, age bands, salary bands, tenure bands, and dependent/employment interactions

## Evaluation Results

Notebook validation performance:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9645 | 0.9842 | 0.9579 | 0.9709 | 0.9939 |
| Regularized Random Forest | 0.9955 | 0.9936 | 0.9992 | 0.9964 | 1.0000 |
| Regularized Gradient Boosting | 0.9995 | 1.0000 | 0.9992 | 0.9996 | 1.0000 |
| SGD Logistic Regression | 0.9460 | 0.9804 | 0.9312 | 0.9551 | 0.9900 |
| Regularized XGBoost | 0.9985 | 1.0000 | 0.9976 | 0.9988 | 1.0000 |



Production training-script validation performance:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9645 | 0.9842 | 0.9579 | 0.9709 | 0.9939 |
| XGBoost | 0.9985 | 1.0000 | 0.9976 | 0.9988 | 1.0000 |

The best validation model is XGBoost by F1 score.

Logistic regression was kept because it gives a transparent benchmark that is easy to explain and debug. XGBoost was chosen as the main final candidate because it achieves near-best performance while handling non-linear feature interactions well, supports regularization, exposes useful feature importance diagnostics, and fits naturally into the same sklearn pipeline and MLflow workflow used by the project.

Saved final model test performance:

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

## High-Score Validity Check

The near-perfect validation and test scores should be treated as a warning sign rather than accepted at face value. The project code does not show obvious classic leakage: the data is split into train, validation, and test rows before preprocessing, `employee_id` and `enrolled` are removed from the feature matrix, and the preprocessing steps are fitted inside sklearn pipelines using the training data.

However, additional diagnostics suggest that the dataset itself is almost perfectly rule-separable. The raw feature combinations show very sharp enrollment patterns:

| Dependents | Employment type | Enrollment rate |
| --- | --- | ---: |
| No | Contract | 0.00% |
| No | Part-time | 0.00% |
| Yes | Full-time | 92.78% |

A shallow diagnostic decision tree trained only on the original non-ID features, without the engineered features used by the production model, reached 0.9995 test accuracy with only one incorrect test prediction. Its decisions were based mainly on salary, age, employment type, and dependent status. A simple rule derived from that tree matched 9,998 of the 10,000 raw labels.

This means the high scores are more likely explained by the target being generated from a simple synthetic or deterministic rule than by the model learning a robust real-world enrollment pattern. XGBoost is probably rediscovering that rule very effectively. The model may be accurate for this specific CSV, but these results should not be interpreted as evidence that it will generalize to real insurance enrollment behavior without validation on an independent dataset.

## Key Takeaways

XGBoost is the strongest model in the production training script, with near-perfect validation and test performance. The model captures the dominant patterns in this dataset very well, especially dependent status, employment type, salary, and age-related patterns.

The logistic regression baseline is also strong, which suggests the dataset has clear separable structure after preprocessing and feature engineering. This is useful because it gives a reliable benchmark and confirms that the preprocessing pipeline is effective.

The very high tree-model scores should be treated carefully. They appear to come from an almost deterministic structure in the dataset, not necessarily from a model that would generalize to real-world enrollment data. Before production use, the model should be evaluated on a genuinely independent sample, preferably from the real data-generating process or from a later time period.
