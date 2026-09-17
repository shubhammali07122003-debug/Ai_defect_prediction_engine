import sqlite3
import os

def setup_database():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/defect_engine.db")
    cursor = conn.cursor()

    # Table 1: Commits (Rohan ke data extraction ke liye)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS commits (
        commit_hash TEXT PRIMARY KEY,
        author_name TEXT,
        commit_date TEXT,
        message TEXT,
        is_bug_fix INTEGER DEFAULT 0
    )
    ''')

    # Table 2: File Changes (Git churn track karne ke liye)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS file_changes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        commit_hash TEXT,
        file_path TEXT,
        lines_added INTEGER,
        lines_deleted INTEGER
    )
    ''')

    # Table 3: Static Code Metrics (Anuska ke static features ke liye)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS static_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT,
        commit_hash TEXT,
        loc INTEGER,
        cyclomatic_complexity REAL
    )
    ''')

    conn.commit()
    conn.close()
    print("Database successfully created: data/defect_engine.db")

if __name__ == "__main__":
    setup_database()