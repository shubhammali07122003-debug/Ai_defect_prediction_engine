import os
import subprocess
import psycopg2
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

def scan_and_refresh_all_metrics(repo_dir: str):
    """
    Recursively scans ALL Python files in the repository and
    refreshes the static_metrics table in Neon DB.
    """
    if not os.path.exists(repo_dir):
        raise FileNotFoundError(f"Target repository not found at: {repo_dir}")

    if not DATABASE_URL:
        raise ValueError("DATABASE_URL missing in .env file.")

    print(f"1. Scanning full repository at '{repo_dir}' for Python source files...")
    all_python_files = []
    
    for root, _, files in os.walk(repo_dir):
        # Skip git internal metadata and virtualenvs
        if ".git" in root or "__pycache__" in root or ".venv" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_dir).replace(os.sep, "/")
                all_python_files.append((rel_path, full_path))

    print(f"Found {len(all_python_files)} total Python files across the codebase.")

    print("2. Connecting to Cloud Neon DB...")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # Table ensure and clear old partial state
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS static_metrics (
            id SERIAL PRIMARY KEY,
            file_path TEXT,
            commit_hash TEXT,
            loc INTEGER,
            cyclomatic_complexity REAL
        );
    """)
    print("Flushing old partial metrics to avoid duplicate / inconsistent split data...")
    cursor.execute("TRUNCATE TABLE static_metrics RESTART IDENTITY;")
    conn.commit()

    print("3. Computing LOC and Cyclomatic Complexity for all files...")
    insert_query = """
        INSERT INTO static_metrics (file_path, commit_hash, loc, cyclomatic_complexity)
        VALUES (%s, %s, %s, %s);
    """

    inserted_count = 0
    for rel_path, full_path in all_python_files:
        metrics = calculate_file_metrics(full_path)
        if metrics:
            commit_hash = get_latest_commit_hash(repo_dir, rel_path)
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

    print("\n--- Neon DB Static Metrics Refreshed Successfully ---")
    print(f"Total files processed and inserted: {inserted_count}")

if __name__ == "__main__":
    TARGET_REPO = "data/raw/flask"
    scan_and_refresh_all_metrics(TARGET_REPO)