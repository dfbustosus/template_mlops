"""Training pipeline orchestration with optional persistence and MLflow tracking."""

from __future__ import annotations

# Predeclare alias for mypy/no-redef stability; assign actual module when available
from typing import Any as _Any
from typing import Dict, Optional

from .data import DataSplit
from .models import ModelResult, evaluate_classifier, save_model, train_classifier

mlflow_module: _Any = None
try:
    import mlflow  # type: ignore

    mlflow_module = mlflow
except Exception:  # pragma: no cover - defensive
    pass


def run_training_pipeline(
    split: DataSplit,
    model,
    params: Dict | None = None,
    persist_path: Optional[str] = None,
    track: bool = False,
) -> ModelResult:
    """Run a minimal training pipeline: train and evaluate.

    Args:
        split: DataSplit containing train/test
        model: an untrained scikit-learn estimator
        params: optional params for model (ignored for generic estimators)
    Returns:
        ModelResult with metrics on the test set
    """
    if track and mlflow_module is not None:
        mlflow_module.start_run()
    clf = train_classifier(model, split.X_train, split.y_train)
    result = evaluate_classifier(clf, split.X_test, split.y_test)
    if persist_path is not None:
        save_model(persist_path, clf)
    if track and mlflow_module is not None:
        try:
            mlflow_module.log_metrics(result.metrics)
            mlflow_module.end_run()
        except Exception:
            # Best-effort logging should not break the pipeline run
            pass
    return result
