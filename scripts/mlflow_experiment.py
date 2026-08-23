import argparse
import sys
import types
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_NAME = "insurance-enrollment-tracking"
TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
ARTIFACT_LOCATION = (PROJECT_ROOT / "mlruns").as_uri()


def setup_mlflow() -> None:
    """Use SQLite for MLflow metadata and mlruns for local artifacts."""

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=ARTIFACT_LOCATION,
        )

    mlflow.set_experiment(EXPERIMENT_NAME)


def tracking_params(
    model_name: str,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build tracking params expected by src.models.train."""

    from src.utils.config import CONFIG

    params = {
        "model_name": model_name,
        "benchmark_model_type": CONFIG.model.benchmark_model_type,
        "final_model_type": CONFIG.model.final_model_type,
        "random_state": CONFIG.data.random_state,
        "threshold": CONFIG.model.threshold,
    }

    if extra_params:
        params.update(extra_params)

    return params


def log_model_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_path: Path,
) -> None:
    """MLflow logger used by the existing src.models.train module."""

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "project": "insurance-enrollment-prediction",
                "training_source": "src.models.train",
                "tracking_backend": "sqlite",
            }
        )
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                f"validation_{metric_name}": float(metric_value)
                for metric_name, metric_value in metrics.items()
            }
        )

        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path), artifact_path="model")


def install_tracking_adapter() -> None:
    """Provide src.utils.tracking without editing the project package."""

    tracking_module = types.ModuleType("src.utils.tracking")
    tracking_module.tracking_params = tracking_params
    tracking_module.log_model_run = log_model_run
    sys.modules["src.utils.tracking"] = tracking_module


def run_experiment() -> None:
    """Train existing project models and track them in MLflow."""

    setup_mlflow()
    install_tracking_adapter()

    from src.models.train import get_best_model, train_models

    results = train_models()
    best_model = get_best_model(results)

    print()
    print("MLflow experiment completed using existing project training code.")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Tracking URI: {TRACKING_URI}")
    print(f"Artifact location: {ARTIFACT_LOCATION}")
    print(f"Best model by validation F1: {best_model.model_name}")

    for result in results.values():
        print()
        print(result.model_name)
        print(result.validation_metrics)
        print(f"saved_to: {result.artifact_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MLflow tracking around the existing project models."
    )
    parser.parse_args()
    run_experiment()


if __name__ == "__main__":
    main()
