from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Main directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
LOGS_DIR = PROJECT_ROOT / "logs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


# Data files
RAW_DATA_PATH = RAW_DATA_DIR / "employee_data.csv"


# Model artifacts
MODEL_PATH = MODELS_DIR / "enrollment_model.joblib"


def create_project_directories() -> None:
    """
    Create directories required by the project if they do not exist.
    """
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        LOGS_DIR,
        MLRUNS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
