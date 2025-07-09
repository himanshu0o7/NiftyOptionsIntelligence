#!/bin/bash
# Error Fix Script: Resolve missing modules for NiftyOptionsIntelligence
# Run this from your project root with venv activated

# Step 1: Ensure venv is active
if [ -z "$VIRTUAL_ENV" ]; then
  echo "❌ Please activate your virtual environment first."
  echo "Use: source venv/bin/activate"
  exit 1
fi

# Step 2: Install required missing packages
REQUIRED_PKGS=(pyotp streamlit pyyaml yfinance pandas plotly requests websockets)

for pkg in "${REQUIRED_PKGS[@]}"; do
  echo "📦 Installing: $pkg"
  pip install "$pkg"
done

# Step 3: Freeze updated requirements
pip freeze > requirements.txt

echo "✅ All missing modules installed. You can now re-run:"
echo "streamlit run app.py --server.port 5000 --server.address 0.0.0.0"

