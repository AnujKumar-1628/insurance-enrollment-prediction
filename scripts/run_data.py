import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import DataLoader


def main():
    loader = DataLoader()

    splits = loader.run()

    print("Train:", splits.X_train.shape)
    print("Validation:", splits.X_validation.shape)
    print("Test:", splits.X_test.shape)
    print("Saved train split:", loader.train_data_path)
    print("Saved validation split:", loader.validation_data_path)
    print("Saved test split:", loader.test_data_path)


if __name__ == "__main__":
    main()
