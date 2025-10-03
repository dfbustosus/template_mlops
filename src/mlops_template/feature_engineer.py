"""Feature engineering class using sklearn transformers.

This class builds a ColumnTransformer pipeline that handles numeric and
categorical columns, with sensible defaults: imputation, scaling for numeric
and one-hot encoding for low-cardinality categoricals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from category_encoders import HashingEncoder
except Exception:  # pragma: no cover - optional dependency
    HashingEncoder = None


@dataclass
class FeatureEngineer:
    """Feature engineering pipeline builder using sklearn transformers.

    The class creates a ColumnTransformer that imputes and scales numeric
    features and encodes categorical features. It provides fit/transform and
    persistence helpers.
    """

    numeric_imputer_strategy: str = "mean"
    categorical_imputer_strategy: str = "most_frequent"
    numeric_scaler: Optional[StandardScaler] = None
    one_hot_handle_unknown: str = "ignore"
    categorical_max_categories_for_onehot: int = 50
    categorical_encoder: str = "onehot"  # options: 'onehot', 'hashing'

    def __post_init__(self) -> None:
        """Initialize internal state after dataclass construction."""
        self._fitted = False
        self._transformer: Optional[ColumnTransformer] = None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        """Return a short representation used for debugging."""
        return (
            "FeatureEngineer(numeric_imputer_strategy="
            f"{self.numeric_imputer_strategy!r}, categorical_encoder="
            f"{self.categorical_encoder!r})"
        )

    def fit(
        self,
        df: pd.DataFrame,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
    ) -> "FeatureEngineer":
        """Fit the transformer on the provided DataFrame.

        If `categorical_cols` or `numeric_cols` are not provided, the method will
        infer types using pandas dtypes.
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        if df.shape[0] == 0:
            raise ValueError("Cannot fit FeatureEngineer on empty DataFrame")

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.numeric_imputer_strategy)),
                ("scaler", StandardScaler()),
            ]
        )

        # Avoid deprecated `sparse` kwarg to keep compatibility across sklearn versions
        # Build categorical transformer depending on selected strategy
        if self.categorical_encoder == "onehot":
            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy=self.categorical_imputer_strategy,
                            fill_value="missing",
                        ),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown=self.one_hot_handle_unknown),
                    ),
                ]
            )
        elif self.categorical_encoder == "hashing":
            if HashingEncoder is None:
                raise RuntimeError(
                    "category_encoders.HashingEncoder is not available; "
                    "install the category_encoders package to use hashing encoding."
                )
            # HashingEncoder expects an array input. We will wrap it to be used in
            # a ColumnTransformer.
            categorical_transformer = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy=self.categorical_imputer_strategy,
                            fill_value="missing",
                        ),
                    ),
                    ("hash", HashingEncoder(n_components=8)),
                ]
            )
        else:
            raise ValueError(
                f"unsupported categorical_encoder: {self.categorical_encoder}"
            )

        transformers = []
        if numeric_cols:
            transformers.append(("num", numeric_transformer, numeric_cols))
        if categorical_cols:
            transformers.append(("cat", categorical_transformer, categorical_cols))

        self._transformer = ColumnTransformer(
            transformers=transformers, remainder="drop"
        )
        # Fit transformer to compute encodings and imputer statistics
        self._transformer.fit(df)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame and return a pandas DataFrame with transformed columns.

        Note: the output column names from ColumnTransformer + OneHotEncoder are
        reconstructed for readability when possible.
        """
        if not self._fitted or self._transformer is None:
            raise RuntimeError(
                "FeatureEngineer must be fitted before calling transform"
            )

        arr = self._transformer.transform(df)
        # Convert sparse output to dense for consistent downstream handling
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        # Try to build output column names
        out_cols: List[str] = []
        for name, transformer, cols in self._transformer.transformers_:
            if name == "remainder":
                continue
            if (
                hasattr(transformer, "named_steps")
                and "onehot" in transformer.named_steps
            ):
                ohe = transformer.named_steps["onehot"]
                # build names using categories_
                for i, col in enumerate(cols):
                    categories = ohe.categories_[i]
                    for cat in categories:
                        out_cols.append(f"{col}__{cat}")
            elif (
                hasattr(transformer, "named_steps")
                and "scaler" in transformer.named_steps
            ):
                for col in cols:
                    out_cols.append(col)
            else:
                for col in cols:
                    out_cols.append(col)

        try:
            return pd.DataFrame(arr, columns=out_cols, index=df.index)
        except Exception:
            # fallback to numpy array if columns mismatch
            return pd.DataFrame(np.asarray(arr), index=df.index)

    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Fit the transformer and return the transformed DataFrame."""
        self.fit(df, **kwargs)
        return self.transform(df)

    def save(self, path: str) -> None:
        """Persist the fitted transformer to disk using joblib."""
        if not self._fitted or self._transformer is None:
            raise RuntimeError("FeatureEngineer must be fitted before saving")
        joblib.dump(self._transformer, path)

    def load(self, path: str) -> "FeatureEngineer":
        """Load a persisted transformer from disk and mark as fitted."""
        self._transformer = joblib.load(path)
        self._fitted = True
        return self
