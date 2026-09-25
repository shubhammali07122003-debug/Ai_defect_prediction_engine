import glob
import os
import sys
import traceback

# 1. Setup exact paths to root, src, and preprocessing folder
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
PREPROCESSING_DIR = os.path.join(SRC_DIR, "preprocessing")

for p in [PROJECT_ROOT, SRC_DIR, PREPROCESSING_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 2. Register module aliases for joblib unpickling
try:
    import src.preprocessing.cleaning as cleaning_mod
    sys.modules['cleaning'] = cleaning_mod
    
    # Expose custom transformers like OutlierCapper if pickled at root level
    if hasattr(cleaning_mod, "OutlierCapper"):
        setattr(sys.modules['__main__'], "OutlierCapper", cleaning_mod.OutlierCapper)
except Exception as alias_err:
    print(f"--> Notice: Custom module alias setup info: {alias_err}")

import joblib
import mlflow.pyfunc
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class PredictRequest(BaseModel):
    loc: float = Field(..., ge=0, example=350.0)
    cyclomatic_complexity: float = Field(..., ge=0, example=15.0)
    churn_30d: float = Field(..., ge=0, example=120.0)
    prior_defects: float = Field(..., ge=0, example=2.0)
    file_path: str = Field(default="src/unknown.py", example="src/auth/jwt.py")


PIPELINE_PATH = os.path.join(PROJECT_ROOT, "artifacts", "cleaning_pipeline.joblib")


def load_preprocessing_pipeline():
    """Loads feature scaler and transformer saved during Phase 2 training."""
    if os.path.exists(PIPELINE_PATH):
        try:
            return joblib.load(PIPELINE_PATH)
        except Exception as e:
            print(f"--> Warning: Failed to load preprocessing pipeline: {e}")
    return None


def get_latest_model():
    """Dynamically finds the latest logged MLflow model artifact."""
    model_dirs = glob.glob(os.path.join(PROJECT_ROOT, "mlruns/1/models/m-*/artifacts"))
    if not model_dirs:
        model_dirs = glob.glob(os.path.join(PROJECT_ROOT, "mlruns/1/*/artifacts/random_forest"))
    if not model_dirs:
        return None

    latest_model_path = max(model_dirs, key=os.path.getmtime)
    try:
        loaded_model = mlflow.pyfunc.load_model(latest_model_path)
        print(f"--> Successfully loaded MLflow model from: {latest_model_path}")
        return loaded_model
    except Exception as e:
        print(f"Error loading MLflow model from {latest_model_path}: {e}")
        return None


# Global instances on startup
pipeline = load_preprocessing_pipeline()
model = get_latest_model()


@router.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "AI Defect Prediction Engine API is Running",
        "model_loaded": model is not None,
        "pipeline_loaded": pipeline is not None,
    }


@router.post("/predict")
def predict_defect(data: PredictRequest):
    try:
        input_dict = {
            "churn_30d": data.churn_30d,
            "prior_defects": data.prior_defects,
            "loc": data.loc,
            "cyclomatic_complexity": data.cyclomatic_complexity,
        }
        input_df = pd.DataFrame([input_dict])

        # Apply Joblib Pipeline transformation if present
        if pipeline is not None:
            try:
                transformed_data = pipeline.transform(input_df)
                input_df = pd.DataFrame(transformed_data, columns=input_df.columns)
            except Exception as pe:
                print(f"--> Warning: Pipeline transform failed ({pe}). Proceeding raw.")

        if model is not None:
            py_model = getattr(model, "_model_impl", None)
            expected_features = None

            if py_model and hasattr(py_model, "python_model"):
                underlying = py_model.python_model
                if hasattr(underlying, "model") and hasattr(underlying.model, "feature_names_in_"):
                    expected_features = list(underlying.model.feature_names_in_)

            if not expected_features and py_model and hasattr(py_model, "sklearn_model"):
                if hasattr(py_model.sklearn_model, "feature_names_in_"):
                    expected_features = list(py_model.sklearn_model.feature_names_in_)

            if expected_features:
                input_df = input_df[expected_features]

            # Extract raw model estimator for base prediction
            raw_model = None
            if py_model and hasattr(py_model, "python_model") and hasattr(py_model.python_model, "model"):
                raw_model = py_model.python_model.model
            elif py_model and hasattr(py_model, "sklearn_model"):
                raw_model = py_model.sklearn_model

            if raw_model and hasattr(raw_model, "predict_proba"):
                prob_array = raw_model.predict_proba(input_df)
                base_prob = float(prob_array[0][1]) if prob_array.ndim == 2 else float(prob_array[0])
            else:
                predictions = model.predict(input_df)
                base_prob = float(predictions[0]) if hasattr(predictions, "__len__") else float(predictions)

            # Risk-calibration boost based on input metrics severity
            complexity_risk = min(0.35, (data.cyclomatic_complexity / 80.0))
            defects_risk = min(0.40, (data.prior_defects * 0.08))
            churn_risk = min(0.15, (data.churn_30d / 300.0))
            loc_risk = min(0.10, (data.loc / 2000.0))

            # Total risk score calculation
            prob = min(0.99, base_prob + complexity_risk + defects_risk + churn_risk + loc_risk)

            is_defective = bool(prob >= 0.50)
            source = "mlflow_model"
        else:
            # Safe Fallback Heuristic
            is_defective = bool(data.loc > 100 or data.cyclomatic_complexity > 10)
            prob = min(1.0, (data.loc * 0.005) + (data.cyclomatic_complexity * 0.05))
            source = "heuristic_fallback"

        return {
            "file_path": data.file_path,
            "is_defective": is_defective,
            "defect_probability": round(prob, 4),
            "input_received": input_dict,
            "inference_source": source,
        }

    except Exception as e:
        print("\n=== PREDICTION ERROR LOG ===")
        traceback.print_exc()
        print("============================\n")
        raise HTTPException(status_code=500, detail=str(e))