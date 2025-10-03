"""Command-line entrypoints for the template."""

from __future__ import annotations

import argparse
from typing import Optional

from sklearn.ensemble import RandomForestClassifier

from .data import load_tabular, train_test_split_df
from .mlflow_utils import configure_local_mlflow
from .pipeline import run_training_pipeline
from .utils import configure_logger


def main(argv: Optional[list] = None) -> int:
    """Command-line entrypoint used by the console script.

    Returns exit code 0 on success.
    """
    parser = argparse.ArgumentParser(prog="mlops-template")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--persist", help="Path to persist trained model (joblib)")
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable MLflow tracking (local file-based by default)",
    )
    args = parser.parse_args(argv)

    logger = configure_logger()
    logger.info("Loading data")
    df = load_tabular(args.data)
    if args.track:
        configure_local_mlflow()
    split = train_test_split_df(df, args.target)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    result = run_training_pipeline(
        split, model, persist_path=args.persist, track=args.track
    )
    logger.info(f"Metrics: {result.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
