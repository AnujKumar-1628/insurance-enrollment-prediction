import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.evaluate import classification_metrics
from src.models.predict import load_model
from src.models.train import split_features_target
from src.utils.config import CONFIG
from src.utils.logger import get_logger
from src.utils.path import FIGURES_DIR


logger = get_logger(__name__)


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the processed test split."""

    test_data = pd.read_csv(CONFIG.data.test_data_path)
    X_test, y_test = split_features_target(test_data)
    logger.info("Loaded test split: %d rows", len(test_data))
    return X_test, y_test


def evaluate_saved_model() -> dict[str, float]:
    """Evaluate the saved final model on the test split."""

    model = load_model(CONFIG.model.final_model_path)
    X_test, y_test = load_test_data()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_test, y_pred, y_proba)
    logger.info("Test metrics: %s", metrics)

    save_confusion_matrix(y_test, y_pred)

    return metrics


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> Path:
    """Create and save the test-set confusion matrix."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "confusion_matrix_test.png"

    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Not enrolled", "Enrolled"],
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved confusion matrix to %s", output_path)
    return output_path


def main() -> None:
    metrics = evaluate_saved_model()

    print("Test metrics")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
