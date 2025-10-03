"""Data schemas using pandera for validation.

Pandera is an optional dependency for this template. If pandera is
installed, `validate_dataframe` will run schema validation. If it is not
installed, attempting to validate will raise a helpful RuntimeError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

# Predeclare pandera alias to avoid duplicate-definition complaints from mypy
# When type checking, import the concrete DataFrameSchema type for static
# analyzers. At runtime we avoid binding the name to different objects which
# can confuse mypy when pre-commit runs across multiple files.
if TYPE_CHECKING:  # pragma: no cover - typing-only
    from pandera import DataFrameSchema  # type: ignore
else:
    DataFrameSchema = Any

# Predeclare pandera alias to avoid duplicate-definition complaints from mypy
_pa: Any = None
try:
    import pandera as pa  # type: ignore

    _pa = pa
except Exception:  # pragma: no cover - optional dependency
    # leave _pa as None
    pass


def validate_dataframe(df: pd.DataFrame, schema: Any) -> pd.DataFrame:
    """Validate a dataframe against a pandera schema and return it if valid.

    If pandera is not installed, raises RuntimeError explaining the dependency.
    """
    if _pa is None:
        raise RuntimeError(
            "pandera is not installed; install pandera to enable schema validation"
        )
    # At this point we expect a pandera DataFrameSchema
    return schema.validate(df)
