"""Property-based fallback tests for data loading and feature engineering.

This module contains property-based tests using Hypothesis with a
deterministic fallback when Hypothesis is not available.
"""

import pandas as pd

from mlops_template.data import DataLoader
from mlops_template.feature_engineer import FeatureEngineer

try:
    from hypothesis import given
    from hypothesis import strategies as st

    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=100),
                st.integers(min_value=0, max_value=100),
            ),
            min_size=2,
            max_size=50,
        )
    )
    def test_dataloader_split_proportion(pairs):
        """Property test: split preserves approximate proportion for random inputs."""
        df = pd.DataFrame(pairs, columns=["a", "y"])  # numeric features, y as int label
        loader = DataLoader()
        split = loader.split(
            df, target_column="y", test_size=0.3, random_state=0, stratify=False
        )
        # check proportions roughly match (allow 1 sample difference)
        total = len(df)
        assert abs(len(split.X_test) - int(total * 0.3)) <= 1

    @given(
        st.lists(
            st.tuples(
                st.floats(allow_nan=False, allow_infinity=False),
            ),
            min_size=2,
            max_size=50,
        )
    )
    def test_feature_engineer_numeric_roundtrip(values):
        """Property test: numeric-only DataFrame roundtrips through FeatureEngineer."""
        # numeric-only DataFrame
        df = pd.DataFrame(values, columns=["x"]).astype(float)
        fe = FeatureEngineer()
        fe.fit(df)
        out = fe.transform(df)
        # transformed rows should match input rows length
        assert len(out) == len(df)

except Exception:
    # Hypothesis not available: run small deterministic tests instead

    def test_dataloader_split_proportion_deterministic():
        """Deterministic fallback for environments without Hypothesis."""
        df = pd.DataFrame({"a": list(range(10)), "y": [0, 1] * 5})
        loader = DataLoader()
        split = loader.split(
            df, target_column="y", test_size=0.3, random_state=0, stratify=False
        )
        total = len(df)
        assert abs(len(split.X_test) - int(total * 0.3)) <= 1

    def test_feature_engineer_numeric_roundtrip_deterministic():
        """Deterministic fallback ensuring FeatureEngineer roundtrip length."""
        df = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        fe = FeatureEngineer()
        fe.fit(df)
        out = fe.transform(df)
        assert len(out) == len(df)
