"""Data loading and dataset utilities.

This module exposes a DataLoader abstraction that handles CSV loading,
validation and safe train/test splitting. The interface is small and
deterministic and intentionally avoids heavy opinionation about validation
rules while providing safe defaults for common edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from .schemas import validate_dataframe
from .settings import settings


@dataclass(frozen=True)
class DataSplit:
    """Typed container holding train/test splits.

    Attributes:
        X_train: training features
        X_test: test features
        y_train: training labels
        y_test: test labels
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class DataLoader:
    """Deterministic data loader with validation and safe splitting.

    Usage example:
        loader = DataLoader()
        df = loader.load_csv(path)
        split = loader.split(df, target_column="y")
    """

    def load_csv(
        self, path: str, required_columns: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """Load CSV and validate required columns.

        Raises:
            FileNotFoundError: if file is missing
            ValueError: if required columns are missing
        """
        df = pd.read_csv(path)
        if required_columns:
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                raise ValueError(f"missing required columns: {missing}")
        return df

    def load_and_validate(
        self,
        path: str,
        schema: Optional[Any] = None,
        required_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Load CSV and optionally validate with a pandera schema."""
        df = self.load_csv(path, required_columns=required_columns)
        if schema is not None:
            validate_dataframe(df, schema)
        return df

    def split(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        stratify: bool = True,
    ) -> DataSplit:
        """Split dataframe into train/test and return a typed DataSplit.

        Args:
            df: input DataFrame
            target_column: name of target column
            test_size: fraction used as test set
            random_state: seed for deterministic splits
            stratify: when True, attempt to stratify splits by the target for
                classification tasks

        Notes:
            If stratify is requested but impossible (for example when there are
            too few samples per class), the function will fall back to a
            non-stratified split to avoid failing the pipeline.
        """
        if target_column not in df.columns:
            raise ValueError(
                f"target_column '{target_column}' not found in dataframe columns"
            )

        if df.shape[0] < 2:
            raise ValueError(
                "dataframe must contain at least 2 rows to perform a split"
            )

        X = df.drop(columns=[target_column])
        y = df[target_column]

        if random_state is None:
            # Use project-level default seed when not provided
            random_state = settings.python_random_seed

        stratify_values = None
        if stratify:
            try:
                # Only use stratify when each class has at least two samples and the
                # test split will contain at least one sample per class.
                class_counts = y.value_counts()
                if class_counts.min() >= 2 and int(len(y) * test_size) >= 1:
                    stratify_values = y
            except Exception:
                # If anything goes wrong computing stratify values, fall back to
                # non-stratified split to avoid failing the whole pipeline.
                stratify_values = None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_values,
        )

        return DataSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def load_tabular(
    csv_path: str, required_columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """Backward-compatible helper to load CSV files.

    Delegates to DataLoader.load_csv.
    """
    loader = DataLoader()
    return loader.load_csv(csv_path, required_columns=required_columns)


def train_test_split_df(
    df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42
) -> DataSplit:
    """Backward-compatible wrapper around DataLoader.split.

    This helper keeps the original function signature expected by older code.
    """
    loader = DataLoader()
    return loader.split(
        df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )
