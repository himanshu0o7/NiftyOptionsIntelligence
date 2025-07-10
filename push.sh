#!/bin/bash
# Script: push.sh
# Purpose: Automate Git pull, commit, and push with timestamp

set -e  # exit on error

# Navigate to project root (optional)
cd /root/NiftyOptionsIntelligence || exit 1

# Pull latest changes first
echo "🔄 Pulling latest changes from main..."
git pull origin main

# Stage all changes
echo "📦 Staging all changes..."
git add .

# Commit with timestamp
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
MESSAGE="🚀 Auto-push on $TIMESTAMP"
echo "📝 Committing with message: $MESSAGE"
git commit -m "$MESSAGE" || echo "⚠️ No changes to commit"

# Push to GitHub
echo "🚀 Pushing to origin/main..."
git push origin main || echo "⚠️ Nothing to push"

echo "✅ Push completed."

