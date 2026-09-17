import os
import psycopg2
import pandas as pd
from radon.raw import analyze
from radon.complexity import cc_visit
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def calculate_file_metrics(file_abs_path: str):
    """
    Computes Lines of Code (LOC) and average Cyclomatic Complexity.
    """
    if not os.path.exists(file_abs_path):
        return None

    try:
        with open(file_abs_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        raw_stats = analyze(code)
        loc = raw_stats.loc

        blocks = cc_visit(code)
        avg_complexity = (sum(b.complexity for b in blocks) / len(blocks)) if blocks else 0.0

        return {
            "loc": int(loc),
            "cyclomatic_complexity": round(float(avg_complexity), 2)
        }
    except Exception as e:
        print(f"Skipping {file_abs_path}: {e}")
        return None

def mine_and_store_static_metrics(repo_base_path: str, csv_path: str):
    """
    Reads git_changes.csv, calculates code metrics, and inserts them
    directly into the live Neon PostgreSQL database.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file changes CSV not found: {csv_path}")

    if not DATABASE_URL:
        raise ValueError("DATABASE_URL not found in .env file.")

    df_changes = pd.read_csv(csv_path)

    latest_file_commits = (
        df_changes.groupby("file_path")
        .first()
        .reset_index()[["file_path", "commit_hash"]]
    )

    print(f"Connecting to Neon Cloud PostgreSQL...")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # Ensure static_metrics table exists in Postgres
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS static_metrics (
            id SERIAL PRIMARY KEY,
            file_path TEXT,
            commit_hash TEXT,
            loc INTEGER,
            cyclomatic_complexity REAL
        );
    """)
    conn.commit()

    print(f"Analyzing static metrics for {len(latest_file_commits)} files...")
    inserted_count = 0

    insert_query = """
        INSERT INTO static_metrics (file_path, commit_hash, loc, cyclomatic_complexity)
        VALUES (%s, %s, %s, %s);
    """

    for _, row in latest_file_commits.iterrows():
        rel_path = row["file_path"]
        commit_hash = row["commit_hash"]
        full_path = os.path.join(repo_base_path, rel_path)

        metrics = calculate_file_metrics(full_path)

        if metrics:
            cursor.execute(insert_query, (
                rel_path,
                commit_hash,
                metrics["loc"],
                metrics["cyclomatic_complexity"]
            ))
            inserted_count += 1

    conn.commit()
    cursor.close()
    conn.close()

    print("\n--- Live Neon DB Sync Completed ---")
    print(f"Rows pushed to Cloud static_metrics: {inserted_count}")

if __name__ == "__main__":
    REPO_DIR = "data/raw/flask"
    CSV_INPUT = "data/processed/git_changes.csv"

    mine_and_store_static_metrics(REPO_DIR, CSV_INPUT)