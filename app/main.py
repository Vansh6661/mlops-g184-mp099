"""FastAPI inference service for Wheat Yield Prediction (Problem MP099)."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field

# 1. Configure structured JSON logging to stdout
logger = logging.getLogger("prediction_service")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


# 2. Define Pydantic request schema with the client's 2 validation rules
class WheatYieldRequest(BaseModel):
    rainfall_mm: float
    fertilizer_kg: float = Field(
        ..., ge=10.0, description="Rule 2: Fertilizer kg below 10 is rejected"
    )
    sowing_delay_days: int
    irrigation_count: int
    soil_ph: float
    farm_acres: float = Field(
        ..., le=25.0, description="Rule 1: Farm acres above 25 is rejected"
    )
    seed_variety: str
    tehsil: str


class PredictionResponse(BaseModel):
    prediction: float


# 3. Initialize FastAPI app
app = FastAPI(
    title="Wheat Yield Prediction API",
    description="Inference service for Bhopal Kisan Producer Company (MP099)",
    version="1.0.0",
)

# 4. Load champion model artifact at startup
MODEL_PATH = Path("models/model.joblib")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    logger.error(json.dumps({"event": "model_load_failure", "error": str(e)}))


# 5. Health check endpoint (Handbook Requirement: status ok)
@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint for Kubernetes liveness/readiness probes."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable")
    return {"status": "ok"}


# 6. Prediction endpoint with client rules & JSON logging
@app.post("/predict", response_model=PredictionResponse)
def predict_yield(request: WheatYieldRequest) -> PredictionResponse:
    """Predict wheat yield per acre and log query/response in JSON format."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable")

    input_data = request.model_dump()
    input_df = pd.DataFrame([input_data])

    # Run inference
    raw_pred = float(model.predict(input_df)[0])

    # Structured JSON log line to stdout
    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": "/predict",
        "input": input_data,
        "prediction": raw_pred,
    }
    logger.info(json.dumps(log_record))

    return PredictionResponse(prediction=raw_pred)
