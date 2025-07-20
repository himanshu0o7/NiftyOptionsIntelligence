#!/bin/bash

# Check if filename is passed
if [ -z "$1" ]; then
  echo "❌ Error: No file passed."
  echo "Usage: ./run_codex_fix.sh <filename>"
  exit 1
fi

FILE_TO_FIX="$1"
CONFIG_FILE="codex_config.json"
COMMIT_MSG="\U0001F680 Auto fix using Codex - Best of 3"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "❌ Error: Config file '$CONFIG_FILE' not found."
  exit 1
fi

echo "🛠 Running Codex Auto Fix for: $FILE_TO_FIX"

# Fix the file using Codex
codex fix "$FILE_TO_FIX" \
  --config "$CONFIG_FILE" \
  --commit "$COMMIT_MSG"

# Check for errors in Codex run
if [ $? -ne 0 ]; then
  echo "❌ Codex fix failed. Check your inputs or Codex login."
  exit 1
fi

# Confirm diff created
echo "✅ Diff created. Review using: codex diff"

