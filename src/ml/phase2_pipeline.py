"""
Phase 2 - Bug Labeling & Baseline Model (matches REAL schema)

Real schema (confirmed via inspect_schema.py against the Neon DB):

file_changes:
    commit_hash, author_name, author_email, committed_date,
    commit_message, file_path, change_type, lines_added,
    lines_deleted, churn

static_metrics:
    id, file_path, commit_hash, loc, cyclomatic_complexity
    (NOTE: no date column -- joined to file_changes via commit_hash
     to get the snapshot's actual date)

There is no explicit is_bug_fix flag, so it's derived from
commit_message using keyword matching -- this is the PRD's "medium"
evidence tier (a commit message that can be reliably classified as a
defect fix). This is a simplification versus "strong" evidence (an
issue-linked bug-fix commit or a PR explicitly tagged as a bug fix),
which would require issue/PR data not present in this schema yet.

Usage:
    python phase2_pipeline.py
"""

import os
import re
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

load_dotenv()

# ============================================================
# CONFIG
# ============================================================
DB_URL = os.getenv("DATABASE_URL")

FILE_CHANGES_TABLE = "file_changes"
FC_FILE_COL = "file_path"
FC_DATE_COL = "committed_date"
FC_ADD_COL = "lines_added"
FC_DEL_COL = "lines_deleted"
FC_MESSAGE_COL = "commit_message"
FC_COMMIT_COL = "commit_hash"

STATIC_METRICS_TABLE = "static_metrics"
SM_FILE_COL = "file_path"
SM_COMMIT_COL = "commit_hash"
SM_FEATURE_COLS = ["loc", "cyclomatic_complexity"]

# Medium-tier evidence per PRD: commit message keywords suggesting a bug fix.
# Document/refine this list with the team -- it directly drives label quality.
BUGFIX_KEYWORDS = [
    "fix", "bug", "issue", "error", "crash", "broken",
    "defect", "patch", "resolve", "regression", "fail",
]
BUGFIX_PATTERN = re.compile(r"\b(" + "|".join(BUGFIX_KEYWORDS) + r")\b", re.IGNORECASE)

CHURN_WINDOW_DAYS = 30
LABEL_HORIZON_DAYS = 30

RANDOM_SEED = 42


def load_data():
    if not DB_URL:
        raise RuntimeError(
            "DATABASE_URL not set. Create a .env file with the connection "
            "string Shubham shared."
        )
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        file_changes = pd.read_sql(f'SELECT * FROM "{FILE_CHANGES_TABLE}"', conn)
        static_metrics = pd.read_sql(f'SELECT * FROM "{STATIC_METRICS_TABLE}"', conn)

    # committed_date has a timezone offset (e.g. "+02:00") -- parse then drop
    # tz so all downstream comparisons use naive timestamps consistently.
    file_changes[FC_DATE_COL] = pd.to_datetime(file_changes[FC_DATE_COL], utc=True).dt.tz_localize(None)

    # Derive the bug-fix flag from the commit message (medium-tier evidence).
    file_changes["is_bug_fix"] = file_changes[FC_MESSAGE_COL].fillna("").apply(
        lambda msg: bool(BUGFIX_PATTERN.search(msg))
    ).astype(int)

    # static_metrics has no date of its own -- join to file_changes on
    # commit_hash (+ file_path, in case a commit touches a file more than once)
    # to attach the actual commit date as the prediction point.
    date_lookup = file_changes[[FC_COMMIT_COL, FC_FILE_COL, FC_DATE_COL]].drop_duplicates()
    static_metrics = static_metrics.merge(
        date_lookup,
        left_on=[SM_COMMIT_COL, SM_FILE_COL],
        right_on=[FC_COMMIT_COL, FC_FILE_COL],
        how="left",
    )
    missing_dates = static_metrics[FC_DATE_COL].isna().sum()
    if missing_dates:
        print(f"WARNING: {missing_dates} static_metrics rows had no matching "
              f"commit in file_changes and will be dropped.")
        static_metrics = static_metrics.dropna(subset=[FC_DATE_COL])

    return file_changes, static_metrics


def compute_churn_30d(file_changes: pd.DataFrame, file_path: str, as_of: pd.Timestamp) -> float:
    """Sum of lines_added+lines_deleted in the CHURN_WINDOW_DAYS before as_of. Past only -> no leakage."""
    window_start = as_of - pd.Timedelta(days=CHURN_WINDOW_DAYS)
    mask = (
        (file_changes[FC_FILE_COL] == file_path)
        & (file_changes[FC_DATE_COL] >= window_start)
        & (file_changes[FC_DATE_COL] < as_of)
    )
    subset = file_changes.loc[mask]
    return (subset[FC_ADD_COL] + subset[FC_DEL_COL]).sum()


def compute_prior_defects(file_changes: pd.DataFrame, file_path: str, as_of: pd.Timestamp) -> int:
    """Count of bug-fix commits touching this file strictly BEFORE as_of. Past only -> no leakage."""
    mask = (
        (file_changes[FC_FILE_COL] == file_path)
        & (file_changes[FC_DATE_COL] < as_of)
        & (file_changes["is_bug_fix"] == 1)
    )
    return int(file_changes.loc[mask].shape[0])


def compute_label(file_changes: pd.DataFrame, file_path: str, as_of: pd.Timestamp) -> int:
    """defective = 1 if a bug-fix commit touches this file within LABEL_HORIZON_DAYS AFTER as_of."""
    window_end = as_of + pd.Timedelta(days=LABEL_HORIZON_DAYS)
    mask = (
        (file_changes[FC_FILE_COL] == file_path)
        & (file_changes[FC_DATE_COL] >= as_of)
        & (file_changes[FC_DATE_COL] < window_end)
        & (file_changes["is_bug_fix"] == 1)
    )
    return int(file_changes.loc[mask].shape[0] > 0)


def build_dataset(file_changes: pd.DataFrame, static_metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per static_metrics snapshot = one prediction point per file."""
    rows = []
    for _, row in static_metrics.iterrows():
        file_path = row[SM_FILE_COL]
        as_of = row[FC_DATE_COL]

        record = {
            "file_path": file_path,
            "prediction_timestamp": as_of,
            "churn_30d": compute_churn_30d(file_changes, file_path, as_of),
            "prior_defects": compute_prior_defects(file_changes, file_path, as_of),
        }
        for col in SM_FEATURE_COLS:
            record[col] = row[col]

        record["defective"] = compute_label(file_changes, file_path, as_of)
        rows.append(record)

    return pd.DataFrame(rows)


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    df_sorted = df.sort_values("prediction_timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_frac))
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
    }


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("defect-prediction-phase2-baseline")

    print("Loading data from cloud database...")
    file_changes, static_metrics = load_data()
    print(f"  file_changes: {len(file_changes)} rows")
    print(f"  static_metrics (with matched dates): {len(static_metrics)} rows")
    print(f"  Commits classified as bug-fix (medium-tier keyword match): "
          f"{file_changes['is_bug_fix'].sum()} / {len(file_changes)} "
          f"({file_changes['is_bug_fix'].mean():.1%})")

    print("\nBuilding leakage-safe dataset...")
    dataset = build_dataset(file_changes, static_metrics)
    print(f"  Built {len(dataset)} rows")
    if len(dataset) > 0:
        print(f"  Defect rate: {dataset['defective'].mean():.2%}")

    if dataset["defective"].nunique() < 2:
        print("\nWARNING: only one class present in the labels. With just "
              "33 static_metrics rows this is expected -- treat this as a "
              "pipeline sanity check, not a real evaluation, until more "
              "static_metrics rows are mined.")

    feature_cols = ["churn_30d", "prior_defects"] + SM_FEATURE_COLS
    train_df, test_df = chronological_split(dataset)

    X_train, y_train = train_df[feature_cols], train_df["defective"]
    X_test, y_test = test_df[feature_cols], test_df["defective"]

    print(f"\nTrain rows: {len(X_train)}  |  Test rows: {len(X_test)}")

    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED
        ),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name}_baseline"):
            model.fit(X_train, y_train)

            if len(X_test) > 0:
                probs = model.predict_proba(X_test)[:, 1]
                preds = (probs >= 0.5).astype(int)
                metrics = evaluate(y_test, preds, probs)
            else:
                metrics = {"precision": float("nan"), "recall": float("nan"),
                           "f1": float("nan"), "roc_auc": float("nan"), "pr_auc": float("nan")}

            mlflow.log_params({
                "model_type": name,
                "label_horizon_days": LABEL_HORIZON_DAYS,
                "churn_window_days": CHURN_WINDOW_DAYS,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            })
            mlflow.log_metrics({k: v for k, v in metrics.items() if not np.isnan(v)})
            mlflow.sklearn.log_model(model, name)

            print(f"\n--- {name} ---")
            for k, v in metrics.items():
                print(f"{k:>10}: {v}")

    print("\nDone. Run 'mlflow ui --backend-store-uri sqlite:///mlflow.db' to view both runs.")


if __name__ == "__main__":
    main()
