"""TrainingPipeline orchestration class.

This class wires together DataLoader, FeatureEngineer and ModelManager to run
deterministic training experiments with clear separation of concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .data import DataLoader
from .feature_engineer import FeatureEngineer
from .models import ModelManager, ModelResult
from .seed import set_seed
from .settings import settings


@dataclass
class TrainingPipelineConfig:
    """Configuration for a TrainingPipeline run."""

    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


class TrainingPipeline:
    """Orchestrate a training run: load, featurize, train, and evaluate."""

    def __init__(
        self,
        config: Optional[TrainingPipelineConfig] = None,
        track_mlflow: bool = False,
    ) -> None:
        """Create a TrainingPipeline with optional config and MLflow tracking."""
        self.config = config or TrainingPipelineConfig()
        self.loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.model_manager = ModelManager(track_mlflow=track_mlflow)

    def run(self, df: pd.DataFrame, target_column: str, model) -> ModelResult:
        """Execute the training pipeline on a DataFrame and return evaluation.

        Args:
            df: full dataset containing features and the target column
            target_column: name of the label column in df
            model: scikit-learn compatible estimator to train
        """
        if df.shape[0] == 0:
            raise ValueError("input dataframe is empty")

        # Set deterministic seed for reproducibility across components
        set_seed(self.config.random_state or settings.python_random_seed)

        split = self.loader.split(
            df,
            target_column=target_column,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=self.config.stratify,
        )

        # Fit feature engineer on training data only
        X_train = split.X_train
        X_test = split.X_test
        self.feature_engineer.fit(X_train)
        X_train_t = self.feature_engineer.transform(X_train)
        X_test_t = self.feature_engineer.transform(X_test)
        # Train and evaluate the provided model
        self.model_manager.train(model, X_train_t, split.y_train)
        result = self.model_manager.evaluate(X_test_t, split.y_test)
        return result
