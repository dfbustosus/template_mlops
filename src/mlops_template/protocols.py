"""Typing protocols used across the package.

This module defines a lightweight Protocol for models that resemble
scikit-learn estimators used in examples and tests.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class ModelProtocol(Protocol):
    """A minimal protocol for scikit-learn like estimators used by this template."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelProtocol":
        """Fit the estimator on (X, y)."""
        ...

    def predict(self, X: pd.DataFrame) -> list:
        """Return predictions for X as a list-like object."""
        ...
