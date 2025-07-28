import streamlit as st

st.set_page_config(page_title="📊 KP5Bot Dashboard", layout="wide")

st.title("📊 KP5Bot - Nifty Options Intelligence")

st.sidebar.page_link("pages/strategy_config.py", label="🧠 Strategy Config")
st.sidebar.page_link("pages/greeks_dashboard.py", label="📈 Greeks Live Feed")
st.sidebar.page_link("pages/signal_tracker.py", label="🔔 Signal Tracker")

st.markdown("Welcome to your trading bot dashboard! Use the sidebar to explore live data feeds, strategy configuration, and signal logs.")

