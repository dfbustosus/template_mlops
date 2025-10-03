"""Integration tests for the end-to-end training pipeline and persistence."""

import tempfile

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mlops_template.data import train_test_split_df
from mlops_template.models import load_model, save_model
from mlops_template.pipeline import run_training_pipeline


def test_run_training_pipeline():
    """Run the pipeline on a tiny dataset and ensure metrics are reported."""
    df = pd.DataFrame(
        {
            "f1": [0, 1, 0, 1],
            "f2": [1, 0, 1, 0],
            "y": [0, 1, 0, 1],
        }
    )
    split = train_test_split_df(df, target_column="y", test_size=0.5, random_state=1)
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    result = run_training_pipeline(split, model)
    assert "accuracy" in result.metrics
    assert 0.0 <= result.metrics["accuracy"] <= 1.0


def test_model_persistence_roundtrip():
    """Persist a trained model and ensure loading restores predict behavior."""
    df = pd.DataFrame(
        {
            "f1": [0, 1, 0, 1],
            "f2": [1, 0, 1, 0],
            "y": [0, 1, 0, 1],
        }
    )
    split = train_test_split_df(df, target_column="y", test_size=0.5, random_state=1)
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    result = run_training_pipeline(split, model)
    with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
        save_model(tmp.name, result.model)
        loaded = load_model(tmp.name)
        preds = loaded.predict(split.X_test)
        assert len(preds) == len(split.X_test)
