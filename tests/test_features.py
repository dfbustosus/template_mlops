"""Tests for simple feature helpers in the `features` module."""

import pandas as pd

from mlops_template.features import create_interaction_feature, select_numeric_features


def test_select_numeric_features():
    """select_numeric_features should return only numeric column names."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    nums = select_numeric_features(df)
    assert nums == ["a"]


def test_create_interaction_feature():
    """create_interaction_feature should add a new column with multiplied values."""
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    out = create_interaction_feature(df, "x", "y", name="z")
    assert "z" in out.columns
    assert out["z"].tolist() == [3, 8]
