"""Data schemas using pandera for validation.

Pandera is an optional dependency for this template. If pandera is
installed, `validate_dataframe` will run schema validation. If it is not
installed, attempting to validate will raise a helpful RuntimeError.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    import pandera as pa  # type: ignore
    from pandera import DataFrameSchema  # type: ignore

    pa_module = pa
except Exception:  # pragma: no cover - optional dependency
    from typing import Any as _Any

    pa_module: _Any = None
    DataFrameSchema = _Any


def validate_dataframe(df: pd.DataFrame, schema: Any) -> pd.DataFrame:
    """Validate a dataframe against a pandera schema and return it if valid.

    If pandera is not installed, raises RuntimeError explaining the dependency.
    """
    if pa_module is None:
        raise RuntimeError(
            "pandera is not installed; install pandera to enable schema validation"
        )
    # At this point we expect a pandera DataFrameSchema
    return schema.validate(df)
