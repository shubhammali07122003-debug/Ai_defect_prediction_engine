import os
import pandas as pd
from pydriller import Repository

def mine_repository_history(repo_path: str, output_csv: str, max_commits: int = None):
    """
    Rohan - Phase 2: High-Speed Git Mining
    Extracts commit hash, author, timestamp, message, lines added/deleted, and churn.
    Complexity extraction deferred to Phase 4 for optimal performance.
    """
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Target repository path not found: {repo_path}")

    records = []
    commit_count = 0
    print(f"Starting optimized mining on: {repo_path}...")

    # Traverse commits without calculating AST complexity per diff
    for commit in Repository(repo_path, only_no_merge=True).traverse_commits():
        commit_count += 1
        commit_hash = commit.hash
        author_name = commit.author.name
        author_email = commit.author.email
        committed_date = commit.committer_date.isoformat()
        first_line_msg = commit.msg.strip().split("\n")[0]

        for modified_file in commit.modified_files:
            file_path = modified_file.new_path or modified_file.old_path
            
            # Fast filter: Python source files only, skipping tests, docs, and build files
            if (
                file_path 
                and file_path.endswith(".py") 
                and not any(ignored in file_path.lower() for ignored in ["test", "tests", "docs", "setup.py"])
            ):
                normalized_path = file_path.replace(os.sep, "/")

                records.append({
                    "commit_hash": commit_hash,
                    "author_name": author_name,
                    "author_email": author_email,
                    "committed_date": committed_date,
                    "commit_message": first_line_msg,
                    "file_path": normalized_path,
                    "change_type": modified_file.change_type.name,
                    "lines_added": modified_file.added_lines,
                    "lines_deleted": modified_file.deleted_lines,
                    "churn": modified_file.added_lines + modified_file.deleted_lines
                })

        if commit_count % 500 == 0:
            print(f"Processed {commit_count} commits ({len(records)} file-changes logged)...")

        if max_commits and commit_count >= max_commits:
            break

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\n--- Phase 2 Mining Finished ---")
    print(f"Total Commits Processed: {commit_count}")
    print(f"Total File-Change Records: {len(df)}")
    print(f"Saved to: {output_csv}")
    return df

if __name__ == "__main__":
    local_repo = "data/raw/flask"
    output_path = "data/processed/git_changes.csv"
    mine_repository_history(local_repo, output_path, max_commits=None)