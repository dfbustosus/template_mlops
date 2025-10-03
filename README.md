# template_mlops

[![PR Lint & Test](https://img.shields.io/github/actions/workflow/status/dfbustosus/template_mlops/pr-lint.yml?branch=main&style=flat-square)](https://github.com/dfbustosus/template_mlops/)
[![pre-commit](https://img.shields.io/badge/pre--commit-configured-blue.svg)](https://github.com/dfbustosus/template_mlops/blob/main/.pre-commit-config.yaml)

Clean, focused MLOps repository template with examples for training, serving,
and CI. This README is short and actionable — everything you need to get
started and to run CI locally.

Table of contents
- Quick start
- Local developer commands
- CI & pre-commit
- Examples: CLI, serving, MLflow
- Tests & property tests
- Troubleshooting

Quick start (Poetry)
1. Install Poetry: https://python-poetry.org/docs/#installation
2. Install dependencies:

```bash
poetry install
```

Local developer commands
- Run unit tests:

```bash
poetry run pytest -q
```

- Format and sort imports:

```bash
poetry run isort --profile black .
poetry run black .
```

- Lint with flake8:

```bash
poetry run flake8 .
```

CI & pre-commit
- CI: `.github/workflows/pr-lint.yml` runs on pull requests and executes
	formatting checks, flake8, mypy (best-effort), and pytest across a Python
	matrix (3.10/3.11/3.12). A dedicated job runs Hypothesis property tests.
- Pre-commit: the repository includes `.pre-commit-config.yaml` (black,
	isort, mypy, basic housekeeping hooks). Install and run locally:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Examples

- CLI example (from this template)

```bash
poetry run python -m mlops_template.cli --data data/train.csv --target y \
	--persist artifacts/model.joblib
```

- Serving example (development only)

```bash
# run in repo root inside poetry shell
poetry run uvicorn serving.app:app --reload --port 8000
```

- MLflow
The codebase contains a guarded helper at
`src/mlops_template/mlflow_utils.py` to configure a local `mlruns` folder.
MLflow is optional and only required if you enable `--track` in the CLI.

Property-based tests
- The project includes Hypothesis tests where available. CI runs property
	tests in a dedicated job. Locally you can run them after installing
	Hypothesis:

```bash
poetry add --dev hypothesis
poetry run pytest -q tests/test_property_based.py
```

Troubleshooting
- If CI fails on formatting: run `isort` and `black` locally and re-run
	pre-commit.
- If tests fail due to optional deps (mlflow, hypothesis, category_encoders):
	these are optional; install them only when needed, or run tests without
	features that require them.

Contributing
- Run formatters and pre-commit before opening a PR. CI will block merges
	that fail formatting, linting, typing, or tests.

License
- MIT
