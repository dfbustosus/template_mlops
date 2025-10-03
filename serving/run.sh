#!/usr/bin/env bash
set -euo pipefail

# Run the FastAPI example app on port 8000
uvicorn mlops_template.serving.app:app --host 0.0.0.0 --port 8000
