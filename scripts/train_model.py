import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.train import get_best_model, train_models


def main() -> None:
    results = train_models()
    best_model = get_best_model(results)

    for result in results.values():
        print(result.model_name)
        print(result.validation_metrics)
        print(f"saved_to: {result.artifact_path}")
        print()

    print(f"best_model: {best_model.model_name}")


if __name__ == "__main__":
    main()
