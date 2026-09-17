"""
Run this FIRST against the real Neon PostgreSQL database to see exactly
what columns Rohan's file_changes and static_metrics tables have.

Setup:
    1. Copy .env.example to .env
    2. Fill in DATABASE_URL with the connection string Shubham gives you
    3. pip install -r requirements.txt   (adds psycopg2-binary, sqlalchemy, python-dotenv)

Usage:
    python inspect_schema.py
"""

import os
import pandas as pd
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")


def inspect_db():
    if not DB_URL:
        print("ERROR: DATABASE_URL not set. Copy .env.example to .env and fill in the connection string.")
        return

    engine = create_engine(DB_URL)
    inspector = inspect(engine)

    tables = inspector.get_table_names()
    print("Tables found:", tables)
    print("=" * 60)

    with engine.connect() as conn:
        for table in tables:
            print(f"\n--- {table} ---")
            cols = inspector.get_columns(table)
            for c in cols:
                print(f"  {c['name']:<25} {c['type']}")

            sample = pd.read_sql(f'SELECT * FROM "{table}" LIMIT 3;', conn)
            print(f"\nSample rows ({len(sample)} shown):")
            print(sample.to_string(index=False))
            print("-" * 60)


if __name__ == "__main__":
    inspect_db()
