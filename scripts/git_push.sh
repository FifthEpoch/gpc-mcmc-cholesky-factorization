#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/git_push.sh "commit message" [branch]
# Example: ./scripts/git_push.sh "update experiment plots" main

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not inside a git repository."
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"commit message\" [branch]"
  exit 1
fi

commit_message="$1"
branch="${2:-$(git rev-parse --abbrev-ref HEAD)}"

echo "Staging changes..."
git add -A

if git diff --cached --quiet; then
  echo "No staged changes to commit."
  exit 0
fi

echo "Committing..."
git commit -m "$commit_message"

echo "Pushing to origin/$branch..."
git push origin "$branch"

echo "Done: pushed commit to origin/$branch"
