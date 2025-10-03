"""Utility helpers for the template project.

These are intentionally small, dependency-free helpers used by the CLI and the
pipeline during examples and tests.
"""

from __future__ import annotations

import logging
from typing import Any, Dict


def configure_logger(
    name: str = "mlops_template", level: int = logging.INFO
) -> logging.Logger:
    """Return a configured stdlib logger.

    Ensures a single StreamHandler is attached to avoid duplicated logs when
    called multiple times.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dict(obj: Any) -> Dict:
    """Return a dict if obj is a dict-like else raise TypeError.

    This is a small validator used in examples where simple config dicts are
    expected.
    """
    if isinstance(obj, dict):
        return obj
    raise TypeError("expected a dict")
