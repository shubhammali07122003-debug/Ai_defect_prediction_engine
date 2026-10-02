import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier

# 1. Neon DB Connection
DATABASE_URL = os.getenv("DATABASE_URL")

def run_automated_retraining():
    if not DATABASE_URL:
        print(" DATABASE_URL environment variable not found.")
        return False

    print("Fetching latest commit & code metrics from Neon PostgreSQL...")
    engine = create_engine(DATABASE_URL)
    
    # Query historical metrics from DB
    query = "SELECT loc, complexity, churn_30d, prior_defects, is_defective FROM repository_metrics;"
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        print(f"❌ Failed to fetch data from DB: {e}")
        return False

    if len(df) < 50:
        print("Insufficient data points for retraining (Minimum 50 required).")
        return False

    # 2. Preprocessing & Feature Matrix
    X = df[['loc', 'complexity', 'churn_30d', 'prior_defects']]
    y = df['is_defective']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Model Training
    print("Retraining LightGBM Classifier...")
    model = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    print(f"New Model Performance -> ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    # 5. Save Model Artifact
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/lgbm_defect_model.pkl"
    joblib.dump(model, model_path)
    print(f"Retrained model artifact saved to {model_path}")
    return True

if __name__ == "__main__":
    run_automated_retraining()