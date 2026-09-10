"""
Toy end-to-end sanity check for the defect-prediction ML pipeline.
Uses a synthetic dataset shaped like our real features (churn, complexity,
contributor count, prior defects) so the mechanics match what we'll do
on real data from Day 6 onward.

Run:
    python toy_pipeline.py
Then check results at http://127.0.0.1:5000 (mlflow ui must be running).
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def make_toy_dataset(n_rows: int = 2000) -> pd.DataFrame:
    """Simulate file-level rows with feature columns resembling our real ones."""
    churn_30d = np.random.exponential(scale=20, size=n_rows)
    complexity = np.random.normal(loc=10, scale=4, size=n_rows).clip(min=1)
    contributors = np.random.poisson(lam=2, size=n_rows) + 1
    prior_defects = np.random.poisson(lam=0.5, size=n_rows)

    # Ground truth signal: higher churn/complexity/prior defects -> more likely defective
    risk_score = (
        0.03 * churn_30d
        + 0.15 * complexity
        + 0.4 * prior_defects
        + np.random.normal(0, 1, size=n_rows)
    )
    threshold = np.percentile(risk_score, 85)  # ~15% positive rate, like real defect data
    defective = (risk_score > threshold).astype(int)

    return pd.DataFrame({
        "churn_30d": churn_30d,
        "complexity": complexity,
        "contributors": contributors,
        "prior_defects": prior_defects,
        "defective": defective,
    })


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    """
    Stand-in for a real chronological split. Real pipeline will split by
    prediction_timestamp, not randomly -- this just tests the mechanics.
    """
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("defect-prediction-toy-check")

    df = make_toy_dataset()
    train_df, test_df = chronological_split(df)

    feature_cols = ["churn_30d", "complexity", "contributors", "prior_defects"]
    X_train, y_train = train_df[feature_cols], train_df["defective"]
    X_test, y_test = test_df[feature_cols], test_df["defective"]

    print(f"Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")
    print(f"Train defect rate: {y_train.mean():.2%}  |  Test defect rate: {y_test.mean():.2%}")

    with mlflow.start_run(run_name="logreg_baseline_toy"):
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        metrics = {
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
            "pr_auc": average_precision_score(y_test, probs),
        }

        mlflow.log_params({"model_type": "LogisticRegression", "class_weight": "balanced"})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print("\n--- Metrics (toy data, just checking mechanics) ---")
        for k, v in metrics.items():
            print(f"{k:>10}: {v:.3f}")

    print("\nDone. Check the MLflow UI to confirm this run was logged.")


if __name__ == "__main__":
    main()
