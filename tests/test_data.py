"""Unit tests for data loading and splitting utilities."""

import pandas as pd

from mlops_template.data import train_test_split_df


def test_train_test_split_df():
    """Verify the backward-compatible train_test_split_df helper."""
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 1, 0, 1], "y": [0, 1, 0, 1]})
    split = train_test_split_df(df, target_column="y", test_size=0.5, random_state=0)
    assert len(split.X_train) == 2
    assert len(split.X_test) == 2
    assert set(split.X_train.columns) == {"a", "b"}
