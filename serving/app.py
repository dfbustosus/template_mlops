"""Minimal FastAPI service to serve model predictions.

This example is intentionally small and should be adapted for production (auth,
rate-limiting, batching, monitoring hooks, etc.).
"""

from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="mlops_template-serving")


class PredictRequest(BaseModel):
    """Request body for prediction containing rows of numeric features."""

    rows: List[List[float]]


@app.on_event("startup")
def load_model():
    """Load model artifact into application state at startup if available."""
    try:
        app.state.model = joblib.load("/app/model.joblib")
    except Exception:
        app.state.model = None


@app.get("/health")
def health():
    """Health endpoint returning service and model load status."""
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    """Run prediction on provided rows and return predictions as JSON."""
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    arr = np.array(req.rows, dtype=float)
    try:
        preds = model.predict(arr)
        return {"predictions": preds.tolist()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
