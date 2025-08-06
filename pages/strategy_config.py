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
from telegram_alerts import send_telegram_alert

st.write("✅ App Loaded")

import json
import pandas as pd
# …the rest of your imports…
from strategies.breakout_strategy import BreakoutStrategy
from strategies.oi_analysis import OIAnalysis
from config.settings import Settings

import streamlit as st

def show_strategy_config():
    """Display strategy configuration page"""

    try:
        st.header("⚙️ Strategy Configuration")

        # Strategy selection and management
        strategy_tab1, strategy_tab2, strategy_tab3 = st.tabs([
            "🎯 Active Strategies",
            "⚙️ Configure Strategy",
            "📊 Backtest Results"
        ])

        with strategy_tab1:
            show_active_strategies()

        with strategy_tab2:
            show_strategy_configuration()

        with strategy_tab3:
            show_backtest_results()
    except Exception as exc:
        tb = traceback.format_exc()
        send_telegram_alert(f"{MODULE_NAME} error: {exc}\nTraceback:\n{tb}")
        st.error("An error occurred while loading the Strategy Configuration page.")

st.set_page_config(
    page_title="Strategy Config",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔧 Strategy Configuration")
# your logic here


