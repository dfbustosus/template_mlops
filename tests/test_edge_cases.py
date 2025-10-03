"""Edge-case tests for loader, feature engineer, and pipeline behaviors."""

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from mlops_template.data import DataLoader
from mlops_template.feature_engineer import FeatureEngineer
from mlops_template.pipeline_manager import TrainingPipeline


def test_loader_missing_target_column():
    """Splitting without the target column should raise ValueError."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.split(df, target_column="y")


def test_loader_small_dataframe_error():
    """Splitting a single-row DataFrame should raise ValueError."""
    df = pd.DataFrame({"a": [1], "y": [0]})
    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.split(df, target_column="y")


def test_feature_engineer_empty_fit_error():
    """Fitting FeatureEngineer on empty DataFrame should raise ValueError."""
    fe = FeatureEngineer()
    with pytest.raises(ValueError):
        fe.fit(pd.DataFrame())


def test_pipeline_small_dataset_stratify_fallback():
    """Pipeline should handle small datasets and fallback from stratify when needed."""
    # two-class dataset where stratify should fallback if too small
    # use 4 samples so the test split is more likely to contain both labels
    df = pd.DataFrame({"f1": [0, 1, 0, 1], "y": [0, 1, 0, 1]})
    pipe = TrainingPipeline()
    model = RandomForestClassifier(n_estimators=2, random_state=0)
    # Should not raise; pipeline will fallback from stratify if unsuitable
    res = pipe.run(df, target_column="y", model=model)
    assert "accuracy" in res.metrics
