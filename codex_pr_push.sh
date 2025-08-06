#!/bin/bash

# 📍 Usage: ./codex_pr_push.sh <filename.py>

if [ -z "$1" ]; then
  echo "⚠️  Usage: ./codex_pr_push.sh <filename.py>"
  exit 1
fi

FILENAME=$1
BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "🌀 Using branch: $BRANCH"
echo "📤 Updating PR for: $FILENAME"

codex pr update \
  --branch="$BRANCH" \
  --description="🔧 Auto Codex fix applied to $FILENAME"

echo "✅ Pull Request updated successfully."

