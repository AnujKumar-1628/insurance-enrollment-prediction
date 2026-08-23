from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.validate_data import validate_employee_data
from src.utils.config import CONFIG
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class DataSplits:
    """Container for leakage-safe train, validation, and test splits."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


class DataLoader:
    """Load, validate, split, and save the enrollment dataset."""

    def __init__(self):
        self.raw_data_path = CONFIG.data.raw_data_path
        self.processed_data_dir = CONFIG.data.processed_data_dir
        self.train_data_path = CONFIG.data.train_data_path
        self.validation_data_path = CONFIG.data.validation_data_path
        self.test_data_path = CONFIG.data.test_data_path
        self.target_column = CONFIG.data.target_column
        self.id_column = CONFIG.data.id_column
        self.test_size = CONFIG.data.test_size
        self.validation_size = CONFIG.data.validation_size
        self.random_state = CONFIG.data.random_state

    def load_data(self) -> pd.DataFrame:
        """Load the dataset from the configured path."""

        logger.info("Loading data from %s", self.raw_data_path)

        data = pd.read_csv(self.raw_data_path)

        logger.info(
            "Data loaded successfully: %d rows, %d columns",
            data.shape[0],
            data.shape[1],
        )

        return data

    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate raw data before creating any train/test split."""

        validate_employee_data(data)
        logger.info("Data validation completed.")

    def split_rows(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split complete rows into train, validation, and test datasets.

        Splitting complete rows first prevents preprocessing leakage. Any
        imputer, scaler, encoder, or feature selector should be fitted only on
        the train split inside an sklearn Pipeline.
        """

        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")

        if self.test_size + self.validation_size >= 1:
            raise ValueError("test_size + validation_size must be less than 1.")

        train_validation, test = train_test_split(
            data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=data[self.target_column],
        )

        validation_ratio = self.validation_size / (1 - self.test_size)

        train, validation = train_test_split(
            train_validation,
            test_size=validation_ratio,
            random_state=self.random_state,
            stratify=train_validation[self.target_column],
        )

        train = train.reset_index(drop=True)
        validation = validation.reset_index(drop=True)
        test = test.reset_index(drop=True)

        logger.info(
            "Data split completed: train=%d, validation=%d, test=%d",
            len(train),
            len(validation),
            len(test),
        )

        return train, validation, test

    def save_splits(self, splits: DataSplits) -> None:
        """Save train, validation, and test row splits as CSV files."""

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        outputs: dict[Path, pd.DataFrame] = {
            self.train_data_path: splits.train,
            self.validation_data_path: splits.validation,
            self.test_data_path: splits.test,
        }

        for path, frame in outputs.items():
            frame.to_csv(path, index=False)
            logger.info("Saved %s rows to %s", len(frame), path)

    def split_data(self, data: pd.DataFrame) -> DataSplits:
        """Create and return leakage-safe splits from validated data."""

        train, validation, test = self.split_rows(data)

        feature_drop_columns = [self.target_column, self.id_column]

        X_train = train.drop(columns=feature_drop_columns)
        X_validation = validation.drop(columns=feature_drop_columns)
        X_test = test.drop(columns=feature_drop_columns)

        y_train = train[self.target_column]
        y_validation = validation[self.target_column]
        y_test = test[self.target_column]

        return DataSplits(
            train=train,
            validation=validation,
            test=test,
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            y_train=y_train,
            y_validation=y_validation,
            y_test=y_test,
        )

    def run(self) -> DataSplits:
        """Run the complete data pipeline and persist split CSV files."""

        set_seed(self.random_state)
        logger.info("Starting data pipeline.")

        data = self.load_data()

        self.validate_data(data)

        splits = self.split_data(data)

        self.save_splits(splits)

        logger.info("Data pipeline completed.")

        return splits
