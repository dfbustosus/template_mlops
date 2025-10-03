"""Simple MLflow configuration helpers for the template."""

from __future__ import annotations

import os
from typing import Any, Optional

# Predeclare alias used across modules so mypy doesn't see multiple
# definitions when modules import/assign to it independently.
mlflow_module: Any = None
try:
    import mlflow  # type: ignore

    mlflow_module = mlflow
except Exception:  # pragma: no cover - defensive
    pass


def configure_local_mlflow(tracking_dir: Optional[str] = None) -> None:
    """Configure MLflow to use a local directory backend if provided.

    Args:
        tracking_dir: local directory for MLflow tracking artifacts
    """
    if tracking_dir is None:
        tracking_dir = os.environ.get("MLFLOW_TRACKING_DIR", "./mlruns")
    uri = f"file:{os.path.abspath(tracking_dir)}"
    if mlflow_module is not None:
        mlflow_module.set_tracking_uri(uri)
