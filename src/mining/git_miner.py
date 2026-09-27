import os
import re
import subprocess
import psycopg2
from psycopg2.extras import execute_batch
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Bug fix keywords regex
BUG_REGEX = re.compile(
    r"\b(fix(es|ed|ing)?|bug(s)?|defect(s)?|patch(es|ed)?|resolve[s]?|closes?)\b|(#\d+)|(issue(s)?\s*#?\d+)",
    re.IGNORECASE
)

def is_bug_commit(message: str) -> int:
    if not message:
        return 0
    return 1 if bool(BUG_REGEX.search(message)) else 0

def mine_git_repository(repo_path: str):
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Repo path not found: {repo_path}")

    print(f"1. Fetching entire full commit history from {repo_path}...")

    # -n limit hata di hai aur --all lagaya hai complete history fetch karne ke liye
    log_cmd = [
        "git", "-C", repo_path, "log",
        "--all",
        "--numstat",
        "--pretty=format:COMMIT_RECORD|||%H|||%an|||%ad|||%s"
    ]

    result = subprocess.run(
        log_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )

    if result.returncode != 0:
        print(f"Git command error: {result.stderr}")
        return pd.DataFrame(), pd.DataFrame()

    lines = result.stdout.splitlines()
    print(f"Total raw lines fetched from Git: {len(lines)}")

    commits_data = {}
    file_changes = []
    current_commit = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("COMMIT_RECORD|||"):
            parts = line.split("|||")
            if len(parts) >= 5:
                commit_hash = parts[1].strip()
                author_name = parts[2].strip()
                commit_date = parts[3].strip()
                message = "|||".join(parts[4:]).strip()

                bug_label = is_bug_commit(message)
                current_commit = {
                    "commit_hash": commit_hash,
                    "author_name": author_name,
                    "commit_date": commit_date,
                    "message": message,
                    "is_bug_fix": bug_label
                }
                commits_data[commit_hash] = current_commit
        else:
            # Numstat line: added \t deleted \t file_path
            if current_commit:
                parts = line.split("\t")
                if len(parts) == 3:
                    added_str, deleted_str, file_path = parts
                    added = int(added_str) if added_str.isdigit() else 0
                    deleted = int(deleted_str) if deleted_str.isdigit() else 0

                    file_changes.append({
                        "commit_hash": current_commit["commit_hash"],
                        "file_path": file_path,
                        "lines_added": added,
                        "lines_deleted": deleted,
                        "churn": added + deleted,
                        "is_bug_fix": current_commit["is_bug_fix"]
                    })

    df_commits = pd.DataFrame(list(commits_data.values()))
    df_changes = pd.DataFrame(file_changes)

    total_commits = len(df_commits)
    total_bug_commits = int(df_commits["is_bug_fix"].sum()) if total_commits > 0 else 0
    defect_rate = (total_bug_commits / total_commits * 100) if total_commits > 0 else 0.0

    print(f"\n--- Mining Summary ---")
    print(f"Total Unique Commits Mined: {total_commits}")
    print(f"Bug-Fix Commits Identified: {total_bug_commits}")
    print(f"Defect Ratio in Commits: {defect_rate:.2f}%")
    print(f"Total File Change Entries: {len(df_changes)}")

    os.makedirs("data/processed", exist_ok=True)
    df_changes.to_csv("data/processed/git_changes.csv", index=False)
    df_commits.to_csv("data/processed/commits_labeled.csv", index=False)
    print("Saved processed CSV files to data/processed/")

    return df_commits, df_changes

def push_to_neon_db(df_commits: pd.DataFrame, df_changes: pd.DataFrame):
    if df_commits.empty:
        print("No commits to push. Skipping database sync.")
        return

    if not DATABASE_URL:
        print("DATABASE_URL missing in .env. Skipping cloud upload.")
        return

    print("\nConnecting to Cloud Neon PostgreSQL...")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # 1. Base tables ensure karein
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS commits (
            commit_hash TEXT PRIMARY KEY,
            author_name TEXT,
            commit_date TEXT,
            message TEXT,
            is_bug_fix INTEGER DEFAULT 0
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS file_changes (
            id SERIAL PRIMARY KEY,
            commit_hash TEXT,
            file_path TEXT,
            lines_added INTEGER,
            lines_deleted INTEGER
        );
    """)

    # 2. Schema sync: Agar is_bug_fix missing hai toh dynamically add karein
    cursor.execute("""
        ALTER TABLE file_changes 
        ADD COLUMN IF NOT EXISTS is_bug_fix INTEGER DEFAULT 0;
    """)
    conn.commit()

    print("Flushing and inserting updated labeled commits into Neon DB...")
    cursor.execute("TRUNCATE TABLE commits CASCADE;")
    cursor.execute("TRUNCATE TABLE file_changes RESTART IDENTITY;")
    conn.commit()

    # 3. Clean strings (null bytes remove karna) & Bulk Insert Commits
    commit_rows = [
        (
            str(r["commit_hash"]).replace("\x00", ""),
            str(r["author_name"]).replace("\x00", "")[:255],
            str(r["commit_date"]).replace("\x00", ""),
            str(r["message"]).replace("\x00", ""),
            int(r["is_bug_fix"])
        )
        for _, r in df_commits.iterrows()
    ]
    execute_batch(cursor, """
        INSERT INTO commits (commit_hash, author_name, commit_date, message, is_bug_fix)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (commit_hash) DO NOTHING;
    """, commit_rows, page_size=1000)

    # 4. Clean strings & Bulk Insert File Changes
    change_rows = [
        (
            str(r["commit_hash"]).replace("\x00", ""),
            str(r["file_path"]).replace("\x00", ""),
            int(r["lines_added"]),
            int(r["lines_deleted"]),
            int(r["is_bug_fix"])
        )
        for _, r in df_changes.iterrows()
    ]
    execute_batch(cursor, """
        INSERT INTO file_changes (commit_hash, file_path, lines_added, lines_deleted, is_bug_fix)
        VALUES (%s, %s, %s, %s, %s);
    """, change_rows, page_size=2000)

    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Commits and File Changes successfully synced to Neon DB!")

if __name__ == "__main__":
    REPO_DIR = "data/raw/flask"
    df_commits, df_changes = mine_git_repository(REPO_DIR)
    push_to_neon_db(df_commits, df_changes)