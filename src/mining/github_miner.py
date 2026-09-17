import os
import requests
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

def fetch_github_issues(repo_owner="pallets", repo_name="flask"):
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    print(f"Fetching issues/PRs for {repo_owner}/{repo_name}...")
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues?state=closed&per_page=100"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"API Error ({response.status_code}): {response.json().get('message', '')}")
        return pd.DataFrame()

    issues_data = response.json()
    extracted_records = []

    for item in issues_data:
        is_pr = "pull_request" in item
        labels = [l["name"].lower() for l in item.get("labels", [])]
        is_bug_label = any(b_tag in labels for b_tag in ["bug", "defect", "fix", "type: bug"])

        extracted_records.append({
            "issue_id": item.get("number"),
            "title": item.get("title"),
            "is_pull_request": is_pr,
            "state": item.get("state"),
            "labels": ",".join(labels),
            "is_bug_issue": is_bug_label,
            "created_at": item.get("created_at"),
            "closed_at": item.get("closed_at")
        })

    return pd.DataFrame(extracted_records)

def sync_issues_to_db():
    df = fetch_github_issues()
    if df.empty:
        print("No issues fetched or rate limit exceeded.")
        return

    print(f"Fetched {len(df)} issues/PRs from GitHub API.")
    engine = create_engine(DATABASE_URL)
    df.to_sql("github_issues", engine, if_exists="replace", index=False)
    print("? Phase 3 Complete: 'github_issues' table successfully live on Cloud PostgreSQL!")

if __name__ == "__main__":
    sync_issues_to_db()
