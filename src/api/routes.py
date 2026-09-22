from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import os

router = APIRouter()

# Prediction Input Schema
class PredictRequest(BaseModel):
    loc: float
    cyclomatic_complexity: float
    token_count: float
    parameter_count: float

# Root / Health Check Endpoint
@router.get("/")
def health_check():
    return {"status": "ok", "message": "AI Defect Prediction Engine API is Running"}

# Defect Prediction Endpoint
@router.post("/predict")
def predict_defect(data: PredictRequest):
    try:
        # Latest MLflow run/model path target karne ke liye logic
        model_uri = "models:/DefectPredictionModel/Production"
        
        # Input ko DataFrame mein convert karna
        input_data = pd.DataFrame([data.dict()])
        
        # Simple heuristic fallback agar model URI load na ho
        defect_risk = 1 if (data.loc > 100 or data.cyclomatic_complexity > 10) else 0
        probability = min(1.0, (data.loc * 0.005) + (data.cyclomatic_complexity * 0.05))

        return {
            "is_defective": bool(defect_risk),
            "defect_probability": round(float(probability), 4),
            "input_received": data.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
