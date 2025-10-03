"""Project settings with optional pydantic support.

This module loads configuration from environment variables. When pydantic
is available it will use BaseSettings; otherwise a small dataclass fallback
is used to avoid introducing a hard dependency for tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SimpleSettings:
    """Lightweight fallback settings read from environment variables."""

    project_name: str = "mlops_template"
    python_random_seed: int = int(os.environ.get("PYTHON_RANDOM_SEED", "42"))
    mlflow_tracking_uri: Optional[str] = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow_experiment: str = os.environ.get("MLFLOW_EXPERIMENT", "default")
    ghcr_repository: Optional[str] = os.environ.get("GHCR_REPOSITORY")


# Default to the simple settings; at runtime try to use pydantic when
# available. `settings` is typed as Any so static analysis is not strict about
# the concrete type and we avoid call-arg errors from different constructors.
settings: Any = SimpleSettings()


try:
    from pydantic import BaseSettings, Field  # type: ignore

    class PydanticSettings(BaseSettings):
        project_name: str = "mlops_template"
        python_random_seed: int = Field(42, env="PYTHON_RANDOM_SEED")
        mlflow_tracking_uri: Optional[str] = Field(None, env="MLFLOW_TRACKING_URI")
        mlflow_experiment: str = Field("default", env="MLFLOW_EXPERIMENT")
        ghcr_repository: Optional[str] = Field(None, env="MLFLOW_REPOSITORY")

        class Config:
            env_file = ".env"

    settings = PydanticSettings()  # type: ignore[call-arg]
except Exception:
    # Keep the SimpleSettings instance
    pass
