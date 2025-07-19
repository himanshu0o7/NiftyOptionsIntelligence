#!/bin/bash
echo "🛠 Running Codex Auto Fix for: $1"

codex fix $1 \
  --config codex_config.json \
  --commit "🚀 Auto fix using Codex - Best of 3"

echo "✅ Diff created. Review using: codex diff"
