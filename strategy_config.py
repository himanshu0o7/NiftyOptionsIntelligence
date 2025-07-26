import streamlit as st

st.title("✅ strategy_config loaded")

def show_strategy_config():
    st.write("✅ App started")

    try:
        from config.settings import Settings
        st.success("✅ Settings imported")
    except Exception as e:
        st.error(f"❌ Failed to import Settings: {e}")



# pages/strategy_config.py

import sys
from pathlib import Path

# Append project root to sys.path so top‑level packages resolve correctly
CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent  # parent of the `pages` folder
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
import pandas as pd
# …the rest of your imports…
from strategies.breakout_strategy import BreakoutStrategy
from strategies.oi_analysis import OIAnalysis
from config.settings import Settings
