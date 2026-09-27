import os
import subprocess
import psycopg2
import pandas as pd
from radon.raw import analyze
from radon.complexity import cc_visit
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_latest_commit_hash(repo_dir: str, file_rel_path: str) -> str:
    """
    Fetches the latest commit hash where the specific file was modified.
    """
    try:
        cmd = ["git", "-C", repo_dir, "log", "-n", "1", "--pretty=format:%H", "--", file_rel_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        commit = result.stdout.strip()
        return commit if commit else "HEAD"
    except Exception:
        return "HEAD"

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

def mine_and_store_static_metrics(repo_base_path: str, csv_path: str = None):
    """
    Reads git_changes.csv or scans repository, calculates code metrics, 
    and inserts them directly into the live Neon PostgreSQL database.
    """
    if not os.path.exists(repo_base_path):
        raise FileNotFoundError(f"Target repository not found at: {repo_base_path}")

    if not DATABASE_URL:
        raise ValueError("DATABASE_URL missing in .env file.")

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
    print("Flushing old static metrics to ensure clean DB state...")
    cursor.execute("TRUNCATE TABLE static_metrics RESTART IDENTITY;")
    conn.commit()

    file_commit_map = {}

    # Prefer reading latest commits from CSV if available (Rohan's pipeline)
    if csv_path and os.path.exists(csv_path):
        print(f"Reading file commit mapping from {csv_path}...")
        df_changes = pd.read_csv(csv_path)
        if "file_path" in df_changes.columns and "commit_hash" in df_changes.columns:
            latest_file_commits = (
                df_changes.groupby("file_path")
                .first()
                .reset_index()[["file_path", "commit_hash"]]
            )
            for _, row in latest_file_commits.iterrows():
                file_commit_map[row["file_path"]] = row["commit_hash"]

    # Fallback to direct repo scan if CSV not provided or missing
    if not file_commit_map:
        print(f"Scanning '{repo_base_path}' for Python files...")
        for root, _, files in os.walk(repo_base_path):
            if ".git" in root or "__pycache__" in root or ".venv" in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, repo_base_path).replace(os.sep, "/")
                    file_commit_map[rel_path] = get_latest_commit_hash(repo_base_path, rel_path)

    print(f"Analyzing static metrics for {len(file_commit_map)} files...")
    inserted_count = 0

    insert_query = """
        INSERT INTO static_metrics (file_path, commit_hash, loc, cyclomatic_complexity)
        VALUES (%s, %s, %s, %s);
    """

    for rel_path, commit_hash in file_commit_map.items():
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