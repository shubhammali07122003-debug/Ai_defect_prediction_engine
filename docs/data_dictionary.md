# Data Dictionary - AI Software Defect Prediction Engine

This document outlines the database schema, features, and labeling rules used in the project.

## 1. Database Schema (`data/defect_engine.db`)

### Table: `commits`
Stores commit metadata mined from target GitHub repositories.

| Column Name | Data Type | Description |
|---|---|---|
| `commit_hash` | TEXT (PK) | Unique SHA hash of the commit |
| `author_name` | TEXT | Name of the commit author |
| `commit_date` | TEXT | ISO timestamp of the commit |
| `message` | TEXT | Commit message log |
| `is_bug_fix` | INTEGER | Binary indicator (1 = bug fix, 0 = regular commit) |
| `linked_issue_id`| TEXT | External issue tracker ID (if linked) |
| `pr_id` | TEXT | Associated Pull Request ID |
| `pr_label` | TEXT | PR classification tag (bug, feature, chore) |

### Table: `file_changes`
Tracks file-level changes per commit.

| Column Name | Data Type | Description |
|---|---|---|
| `id` | INTEGER (PK)| Auto-incrementing primary key |
| `commit_hash` | TEXT (FK) | Reference to `commits.commit_hash` |
| `file_path` | TEXT | Relative path of the modified file |
| `lines_added` | INTEGER | Total lines inserted |
| `lines_deleted`| INTEGER | Total lines removed |

### Table: `static_metrics`
Contains static code metrics extracted using Lizard.

| Column Name | Data Type | Description |
|---|---|---|
| `id` | INTEGER (PK)| Auto-incrementing primary key |
| `file_path` | TEXT | Path of the analyzed code file |
| `commit_hash` | TEXT (FK) | Reference to `commits.commit_hash` |
| `loc` | INTEGER | Lines of Code |
| `cyclomatic_complexity` | REAL | Average Cyclomatic Complexity score |

---

## 2. Defect Labeling Strategy (30-Day Window)

Target variable `defective` is assigned per file per 30-day future window:
- **Strong Signal**: File modified by an issue-linked bug fix commit OR PR tagged as `bug`.
- **Medium Signal**: File modified by a commit whose message matches bug-fix NLP patterns.
- **Otherwise**: `defective = 0`.
