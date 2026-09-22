"""
Diagnoses why static_metrics rows aren't matching file_changes on
(commit_hash, file_path). Run this after phase2_pipeline.py reports
dropped rows.

Usage:
    python diagnose_join.py
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")


def main():
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        file_changes = pd.read_sql('SELECT commit_hash, file_path FROM file_changes', conn)
        static_metrics = pd.read_sql('SELECT commit_hash, file_path FROM static_metrics', conn)

    fc_hashes = set(file_changes["commit_hash"])
    sm_hashes = set(static_metrics["commit_hash"])

    missing_hashes = sm_hashes - fc_hashes
    print(f"static_metrics has {len(sm_hashes)} distinct commit_hashes")
    print(f"file_changes has {len(fc_hashes)} distinct commit_hashes")
    print(f"commit_hashes in static_metrics but NOT in file_changes: {len(missing_hashes)}")

    if missing_hashes:
        print("\nSample missing commit_hashes:")
        for h in list(missing_hashes)[:5]:
            print(f"  {h}")

    # Check if it's a file_path mismatch instead (hash exists, but path differs)
    merged_on_hash_only = static_metrics.merge(
        file_changes, on="commit_hash", how="left", suffixes=("_sm", "_fc")
    )
    path_mismatch = merged_on_hash_only[
        merged_on_hash_only["file_path_fc"].notna()
        & (merged_on_hash_only["file_path_sm"] != merged_on_hash_only["file_path_fc"])
    ]
    exact_match = static_metrics.merge(
        file_changes, on=["commit_hash", "file_path"], how="inner"
    )

    print(f"\nRows matching on commit_hash AND file_path exactly: {len(exact_match)}")
    print(f"Rows where commit_hash matches but file_path differs (sample): {len(path_mismatch)}")
    if len(path_mismatch) > 0:
        print(path_mismatch[["commit_hash", "file_path_sm", "file_path_fc"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
