"""Project settings with optional pydantic support.

This module loads configuration from environment variables. When pydantic
is available it will use BaseSettings; otherwise a small dataclass fallback
is used to avoid introducing a hard dependency for tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    # pydantic v1 exposes BaseSettings; v2 moved BaseSettings to pydantic-settings
    from pydantic import BaseSettings, Field  # type: ignore

    class Settings(BaseSettings):
        """Pydantic-backed settings object used when pydantic is available."""

        project_name: str = "mlops_template"
        python_random_seed: int = Field(42, env="PYTHON_RANDOM_SEED")
        mlflow_tracking_uri: Optional[str] = Field(None, env="MLFLOW_TRACKING_URI")
        mlflow_experiment: str = Field("default", env="MLFLOW_EXPERIMENT")
        ghcr_repository: Optional[str] = Field(None, env="GHCR_REPOSITORY")

        class Config:
            """Pydantic settings config: load values from .env when present."""

            env_file = ".env"

    settings = Settings()
except Exception:
    # Lightweight fallback to avoid hard dependency on pydantic for tests
    @dataclass
    class SimpleSettings:
        """Lightweight fallback settings read from environment variables."""

        project_name: str = "mlops_template"
        python_random_seed: int = int(os.environ.get("PYTHON_RANDOM_SEED", "42"))
        mlflow_tracking_uri: Optional[str] = os.environ.get("MLFLOW_TRACKING_URI")
        mlflow_experiment: str = os.environ.get("MLFLOW_EXPERIMENT", "default")
        ghcr_repository: Optional[str] = os.environ.get("GHCR_REPOSITORY")

    settings = SimpleSettings()
