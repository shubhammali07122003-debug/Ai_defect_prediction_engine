import os
import re
from collections import defaultdict
from pydriller import Repository

def inspect_repository(repo_path: str, max_commits: int = 300):
    """
    Rohan - Phase 1: Git repository check & mining qualification.
    Validates commit traversal, path parsing, and bug evidence signals.
    """
    if not os.path.exists(repo_path):
        print(f"Error: Directory '{repo_path}' does not exist.")
        return

    commit_count = 0
    file_change_counts = defaultdict(int)
    bug_fix_hints = 0
    bug_pattern = re.compile(r'\b(fix|fixes|fixed|bug|defect|patch|resolve|resolves|issue)\b', re.IGNORECASE)

    print(f"Inspecting repository at: {repo_path}")

    for commit in Repository(repo_path, only_no_merge=True).traverse_commits():
        commit_count += 1

        if bug_pattern.search(commit.msg):
            bug_fix_hints += 1

        for modified_file in commit.modified_files:
            norm_path = modified_file.new_path or modified_file.old_path
            if norm_path and norm_path.endswith('.py'):
                norm_path = norm_path.replace(os.sep, "/")
                file_change_counts[norm_path] += 1

        if max_commits and commit_count >= max_commits:
            break

    print(f"\n--- Phase 1 Qualification Summary ---")
    print(f"Parsed Commits: {commit_count}")
    print(f"Bug-Related Commits: {bug_fix_hints} ({(bug_fix_hints/commit_count)*100:.2f}%)")
    print(f"Unique Python Files Touched: {len(file_change_counts)}")
    print("\nTop 5 High-Churn Python Files:")
    for path, count in sorted(file_change_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {path}: {count} touches")

if __name__ == "__main__":
    local_repo_path = "data/raw/flask"
    inspect_repository(local_repo_path, max_commits=300)