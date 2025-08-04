#!/bin/bash
set -euo pipefail

send_alert() {
  python - <<'PYTHON' "$1"
from telegram_alerts import send_telegram_alert
import sys


def main():
    message = sys.argv[1]
    try:
        send_telegram_alert(message)
    except Exception as exc:  # pragma: no cover
        print(f"Telegram send failed: {exc}")


if __name__ == "__main__":
    main()
PYTHON
}

if [ -z "${1:-}" ]; then
  echo "❌ Error: No file passed."
  send_alert "❌ Codex fix: No file provided."
  echo "Usage: ./run_codex_fix.sh <filename>"
  exit 1
fi

FILE_TO_FIX="$1"
CONFIG_FILE="codex_config.json"
COMMIT_MSG="🚀 Auto fix using Codex - Best of 3"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "❌ Error: Config file '$CONFIG_FILE' not found."
  send_alert "❌ Codex fix: config file missing."
  exit 1
fi

echo "🛠 Running Codex Auto Fix for: $FILE_TO_FIX"

if codex fix "$FILE_TO_FIX" --config "$CONFIG_FILE" --commit "$COMMIT_MSG"; then
  echo "✅ Diff created. Review using: codex diff"
  send_alert "✅ Codex fix applied for $FILE_TO_FIX"
else
  echo "❌ Codex fix failed. Check your inputs or Codex login."
  send_alert "❌ Codex fix failed for $FILE_TO_FIX"
  exit 1
fi

