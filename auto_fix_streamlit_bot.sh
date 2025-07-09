#!/bin/bash
# Script: auto_fix_streamlit_bot.sh
# Purpose: Automatically fix missing modules and repo structure issues for Streamlit bot

set -e  # exit on error

echo "✅ Activating virtual environment..."
source venv/bin/activate

# Step 1: Install all likely missing packages
REQUIRED_PACKAGES=(pyyaml pandas yfinance plotly websockets)

for pkg in "${REQUIRED_PACKAGES[@]}"; do
  if ! pip show "$pkg" > /dev/null 2>&1; then
    echo "📦 Installing: $pkg"
    pip install "$pkg"
  else
    echo "✅ Already installed: $pkg"
  fi
done

# Step 2: Generate requirements.txt
echo "📝 Generating requirements.txt..."
pip freeze > requirements.txt

# Step 3: Add __init__.py to missing folders
MODULE_DIRS=(core strategies utils risk_management pages data)
for dir in "${MODULE_DIRS[@]}"; do
  if [ -d "$dir" ] && [ ! -f "$dir/__init__.py" ]; then
    touch "$dir/__init__.py"
    echo "📁 Added: $dir/__init__.py"
  fi
done

# Step 4: Final check
echo "✅ Setup complete. Run your app with:"
echo "streamlit run app.py --server.port 5000 --server.address 0.0.0.0"

