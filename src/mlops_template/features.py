"""Feature engineering helpers.

Small, pure-Pandas utilities used as illustration for feature transformations.
"""

from __future__ import annotations

from typing import List

import pandas as pd


def select_numeric_features(df: pd.DataFrame) -> List[str]:
    """Return list of numeric feature names.

    Args:
        df: DataFrame

    Returns:
        names of numeric columns
    """
    return df.select_dtypes(include=["number"]).columns.tolist()


def create_interaction_feature(
    df: pd.DataFrame, a: str, b: str, name: str | None = None
) -> pd.DataFrame:
    """Create a new feature that multiplies two existing numeric features.

    This function validates that the input columns exist and returns a copy
    of the DataFrame with the new feature appended.
    """
    if name is None:
        name = f"{a}_x_{b}"

    missing = [c for c in (a, b) if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found in dataframe: {missing}")

    out = df.copy()
    out[name] = out[a] * out[b]
    return out
