from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import glob
import os
import traceback

router = APIRouter()

class PredictRequest(BaseModel):
    loc: float
    cyclomatic_complexity: float
    churn_30d: float
    prior_defects: float

def get_latest_model():
    model_dirs = glob.glob("mlruns/1/models/m-*/artifacts")
    if not model_dirs:
        return None
    
    latest_model_path = max(model_dirs, key=os.path.getmtime)
    try:
        loaded_model = mlflow.pyfunc.load_model(latest_model_path)
        print(f"--> Successfully loaded MLflow model from: {latest_model_path}")
        return loaded_model
    except Exception as e:
        print(f"Error loading MLflow model: {e}")
        return None

model = get_latest_model()

@router.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "AI Defect Prediction Engine API is Running",
        "model_loaded": model is not None
    }

@router.post("/predict")
def predict_defect(data: PredictRequest):
    try:
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])
        
        if model is not None:
            # Unwrap underlying python model
            py_model = getattr(model, "_model_impl", None)
            expected_features = None

            # Check for feature names in various underlying model structures
            if py_model and hasattr(py_model, "python_model"):
                underlying = py_model.python_model
                if hasattr(underlying, "model") and hasattr(underlying.model, "feature_names_in_"):
                    expected_features = list(underlying.model.feature_names_in_)
            
            if not expected_features and py_model and hasattr(py_model, "sklearn_model"):
                if hasattr(py_model.sklearn_model, "feature_names_in_"):
                    expected_features = list(py_model.sklearn_model.feature_names_in_)

            if expected_features:
                print(f"--> Reordering columns to model fit order: {expected_features}")
                input_df = input_df[expected_features]
            else:
                print("--> Warning: Could not extract exact feature order automatically. Passing DataFrame as-is.")

            predictions = model.predict(input_df)
            prob = float(predictions[0]) if hasattr(predictions, "__len__") else float(predictions)
            is_defective = bool(prob >= 0.5)
        else:
            is_defective = bool(data.loc > 100 or data.cyclomatic_complexity > 10)
            prob = min(1.0, (data.loc * 0.005) + (data.cyclomatic_complexity * 0.05))

        return {
            "is_defective": is_defective,
            "defect_probability": round(prob, 4),
            "input_received": input_dict,
            "inference_source": "mlflow_model" if model is not None else "heuristic_fallback"
        }
    except Exception as e:
        print("\n=== PREDICTION ERROR LOG ===")
        traceback.print_exc()
        print("============================\n")
        raise HTTPException(status_code=500, detail=str(e))
