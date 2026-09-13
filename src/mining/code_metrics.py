import os
import sqlite3
import pandas as pd
from radon.raw import analyze
from radon.complexity import cc_visit

def calculate_file_metrics(file_abs_path: str):
    """
    Computes Lines of Code (LOC) and average Cyclomatic Complexity.
    """
    if not os.path.exists(file_abs_path):
        return None

    try:
        with open(file_abs_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        # 1. Calculate LOC (Lines of Code)
        raw_stats = analyze(code)
        loc = raw_stats.loc

        # 2. Calculate Cyclomatic Complexity
        blocks = cc_visit(code)
        if blocks:
            avg_complexity = sum(b.complexity for b in blocks) / len(blocks)
        else:
            avg_complexity = 0.0

        return {
            "loc": int(loc),
            "cyclomatic_complexity": round(float(avg_complexity), 2)
        }
    except Exception as e:
        print(f"Skipping {file_abs_path} due to error: {e}")
        return None

def mine_and_store_static_metrics(repo_base_path: str, csv_path: str, db_path: str):
    """
    Reads git_changes.csv, calculates static metrics for touched files,
    and inserts records into the static_metrics table.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file changes CSV not found: {csv_path}")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Run database.py first.")

    df_changes = pd.read_csv(csv_path)

    # Latest commit hash per file nikalte hain taaki valid commit link rahe
    latest_file_commits = (
        df_changes.groupby("file_path")
        .first()
        .reset_index()[["file_path", "commit_hash"]]
    )

    print(f"Analyzing static metrics for {len(latest_file_commits)} unique files...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    inserted_count = 0

    for _, row in latest_file_commits.iterrows():
        rel_path = row["file_path"]
        commit_hash = row["commit_hash"]
        full_path = os.path.join(repo_base_path, rel_path)

        metrics = calculate_file_metrics(full_path)

        if metrics:
            cursor.execute("""
                INSERT INTO static_metrics (file_path, commit_hash, loc, cyclomatic_complexity)
                VALUES (?, ?, ?, ?)
            """, (
                rel_path,
                commit_hash,
                metrics["loc"],
                metrics["cyclomatic_complexity"]
            ))
            inserted_count += 1

    conn.commit()
    conn.close()

    print("\n--- Code Metrics Mining Completed ---")
    print(f"Rows inserted into static_metrics table: {inserted_count}")
    print(f"Target Database: {db_path}")

if __name__ == "__main__":
    REPO_DIR = "data/raw/flask"
    CSV_INPUT = "data/processed/git_changes.csv"
    DB_PATH = "data/defect_engine.db"

    mine_and_store_static_metrics(REPO_DIR, CSV_INPUT, DB_PATH)