"""Simple MLflow configuration helpers for the template."""

from __future__ import annotations

import os
from typing import Optional

from typing import Any

try:
    import mlflow
except Exception:  # pragma: no cover - defensive
    mlflow: Any = None


def configure_local_mlflow(tracking_dir: Optional[str] = None) -> None:
    """Configure MLflow to use a local directory backend if provided.

    Args:
        tracking_dir: local directory for MLflow tracking artifacts
    """
    if tracking_dir is None:
        tracking_dir = os.environ.get("MLFLOW_TRACKING_DIR", "./mlruns")
    uri = f"file:{os.path.abspath(tracking_dir)}"
    mlflow.set_tracking_uri(uri)
