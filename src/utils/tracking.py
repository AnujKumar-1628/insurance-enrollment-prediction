from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from src.utils.config import CONFIG
from src.utils.logger import get_logger
from src.utils.path import MLRUNS_DIR


logger = get_logger(__name__)


def is_package_available(package_name: str) -> bool:
    """Return whether an optional tracking package is installed."""

    return find_spec(package_name) is not None


def tracking_params(model_name: str, extra_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build common tracking parameters."""

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


def log_to_mlflow(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_path: Path,
) -> None:
    """Log one model run to local MLflow tracking."""

    if not CONFIG.tracking.enable_mlflow:
        logger.info("MLflow tracking disabled by config.")
        return

    if not is_package_available("mlflow"):
        logger.warning("MLflow is not installed. Skipping MLflow tracking.")
        return

    mlflow = import_module("mlflow")

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(CONFIG.tracking.mlflow_tracking_uri)
    mlflow.set_experiment(CONFIG.tracking.experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(artifact_path))

    logger.info("Logged %s to MLflow at %s", run_name, CONFIG.tracking.mlflow_tracking_uri)


def log_to_wandb(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_path: Path,
) -> None:
    """Log one model run to Weights & Biases."""

    if not CONFIG.tracking.enable_wandb:
        logger.info("Weights & Biases tracking disabled by config.")
        return

    if not is_package_available("wandb"):
        logger.warning("Weights & Biases is not installed. Skipping W&B tracking.")
        return

    wandb = import_module("wandb")

    run = wandb.init(
        project=CONFIG.tracking.wandb_project,
        name=run_name,
        config=params,
        mode=CONFIG.tracking.wandb_mode,
        reinit=True,
    )
    run.log(metrics)
    artifact = wandb.Artifact(name=run_name, type="model")
    artifact.add_file(str(artifact_path))
    run.log_artifact(artifact)
    run.finish()

    logger.info("Logged %s to Weights & Biases project %s", run_name, CONFIG.tracking.wandb_project)


def log_model_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_path: Path,
) -> None:
    """Log one model run to all configured experiment trackers."""

    log_to_mlflow(run_name, params, metrics, artifact_path)
    log_to_wandb(run_name, params, metrics, artifact_path)
