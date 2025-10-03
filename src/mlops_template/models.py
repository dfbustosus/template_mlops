"""Model training, evaluation and persistence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

# mlflow is optional for running tests locally; guard import
try:
    import mlflow
except Exception:  # pragma: no cover - defensive
    mlflow = None


@dataclass
class ModelResult:
    """Container for a trained model and computed metrics."""

    model: ClassifierMixin
    metrics: Dict[str, float]


def train_classifier(
    model: ClassifierMixin, X: pd.DataFrame, y: pd.Series
) -> ClassifierMixin:
    """Train a scikit-learn classifier and return it.

    Args:
        model: scikit-learn estimator implementing fit
        X: training features
        y: training labels
    """
    model.fit(X, y)
    return model


def evaluate_classifier(
    model: ClassifierMixin, X: pd.DataFrame, y: pd.Series
) -> ModelResult:
    """Evaluate a trained classifier and return a ModelResult.

    Computes accuracy and, when available, precision, recall and AUC.
    """
    preds = model.predict(X)
    acc = float(accuracy_score(y, preds))
    metrics: Dict[str, float] = {"accuracy": acc}
    try:
        metrics["precision"] = float(precision_score(y, preds, zero_division=0))
        metrics["recall"] = float(recall_score(y, preds, zero_division=0))
    except Exception:
        pass

    try:
        if hasattr(model, "predict_proba") and len(set(y.tolist())) == 2:
            probs = model.predict_proba(X)[:, 1]
            metrics["auc"] = float(roc_auc_score(y, probs))
    except Exception:
        pass

    # Try to include confusion matrix stats
    try:
        # Build explicit labels list for confusion_matrix to avoid sklearn warnings
        y_list = list(y.tolist())
        preds_list = preds.tolist() if hasattr(preds, "tolist") else list(preds)
        labels = sorted(set(y_list + preds_list))
        # sklearn may warn when there is only a single label present; suppress
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="A single label was found in 'y_true' and 'y_pred'.*",
                category=UserWarning,
            )
            cm = confusion_matrix(y, preds, labels=labels)
        # flatten cm into numbers
        metrics["confusion_matrix_sum"] = float(cm.sum())
    except Exception:
        pass

    return ModelResult(model=model, metrics=metrics)


def save_model(joblib_path: str, model: ClassifierMixin) -> None:
    """Persist a model to disk using joblib and track it with MLflow as an artifact."""
    joblib.dump(model, joblib_path)
    # Only attempt to log artifact if mlflow is available and configured
    if mlflow is not None:
        try:
            mlflow.log_artifact(joblib_path)
        except Exception:
            # Do not let MLflow logging break core functionality
            pass


def load_model(joblib_path: str) -> ClassifierMixin:
    """Load a model previously persisted with joblib."""
    return joblib.load(joblib_path)


class ModelManager:
    """Encapsulate training, evaluation, persistence and optional MLflow tracking.

    This class is a small facade over the helper functions above and provides a
    convenient stateful API for running experiments in scripts or tests.
    """

    def __init__(self, track_mlflow: bool = False) -> None:
        """Create a ModelManager.

        Args:
            track_mlflow: when True, attempt to log metrics/artifacts to MLflow if
                the mlflow package is available.
        """
        self.track_mlflow = track_mlflow and mlflow is not None
        self.model: ClassifierMixin | None = None

    def train(
        self, model: ClassifierMixin, X: pd.DataFrame, y: pd.Series
    ) -> ClassifierMixin:
        """Train the provided model on (X, y) and store it internally."""
        trained = train_classifier(model, X, y)
        self.model = trained
        return trained

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelResult:
        """Evaluate the stored model on (X, y) and optionally log to MLflow."""
        if self.model is None:
            raise RuntimeError("no trained model available; call train() first")
        result = evaluate_classifier(self.model, X, y)
        if self.track_mlflow and mlflow is not None:
            try:
                mlflow.log_metrics(result.metrics)
            except Exception:
                pass
        return result

    def save(self, path: str) -> None:
        """Persist the stored model to disk and write a simple signature file."""
        if self.model is None:
            raise RuntimeError("no trained model available to save")
        save_model(path, self.model)
        # write a model signature (feature names if available)
        signature_path = f"{path}.signature.json"
        try:
            feature_names = getattr(self.model, "feature_names_in_", None)
            signature = {
                "feature_names": (
                    list(feature_names) if feature_names is not None else []
                )
            }
            with open(signature_path, "w") as fh:
                json.dump(signature, fh)
        except Exception:
            signature_path = None

        if self.track_mlflow and mlflow is not None:
            try:
                mlflow.log_artifact(path)
                if signature_path is not None:
                    mlflow.log_artifact(signature_path)
            except Exception:
                pass

    def load(self, path: str) -> ClassifierMixin:
        """Load a model from disk and store it internally."""
        self.model = load_model(path)
        return self.model
