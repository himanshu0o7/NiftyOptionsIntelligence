# app.py - Fixed and cleaned version

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import time
import threading
import subprocess
import signal

# Local module imports
from session_manager import SessionManager
from option_stream_ui import get_option_data

# ---------------------------
# PAGE SETUP
# ---------------------------
st.set_page_config(layout="wide")
st_autorefresh(interval=5000, limit=100, key="refresh_autokey_001")
st.title("📈 Nifty Options Intelligence Dashboard")
st.markdown("Use this app to monitor live CE/PE option data for NIFTY/BANKNIFTY")

# ---------------------------
# GLOBALS
# ---------------------------
tokens = None
last_login_time = 0

# ---------------------------
# TOKEN REFRESH FUNCTION
# ---------------------------
def ensure_tokens_fresh():
    global tokens, last_login_time
    if time.time() - last_login_time > (14 * 60):
        time.sleep(1)
        sm = SessionManager()
        session = sm.get_session()
        tokens = session
        last_login_time = time.time()

# ---------------------------
# STREAMLIT UI
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])
    option_type = st.radio("Option Type", ["CE", "PE"], horizontal=True)
with col2:
    strike_price = st.number_input("Select Strike Price", min_value=10000, max_value=50000, step=50, value=22500)

# ---------------------------
# DATA FETCH + DISPLAY
# ---------------------------
try:
    ensure_tokens_fresh()
    data = get_option_data(symbol, strike_price, option_type)
    if data and "error" not in data:
        st.subheader(f"📊 Live Data for {symbol} {strike_price} {option_type}")
        st.dataframe(pd.DataFrame([data]))
    else:
        st.error(data.get("error", "Unknown error occurred"))
except Exception as e:
    st.error(f"⚠️ App error: {e}")

# ---------------------------
# SIGNAL HANDLING (OPTIONAL)
# ---------------------------
try:
    if threading.current_thread() is threading.main_thread():
        def handler(signum, frame):
            print(f"Signal {signum} received")
        signal.signal(signal.SIGTERM, handler)
except Exception as err:
    print(f"[Signal Handling Skipped] Reason: {err}")

# ---------------------------
# START WEBSOCKET
# ---------------------------
if st.button("Start Live WebSocket"):
    subprocess.Popen(["python3", "websocket_runner.py"])
    st.success("WebSocket started in background.")


# app.py

from utils.trend_detector import detect_trend
import streamlit as st

st.subheader("📈 Market Trend Detector")

symbol = st.selectbox("Choose Symbol", ["NIFTY", "BANKNIFTY"])
expiry = st.text_input("Enter Expiry (e.g., 25JUL2025)")

if st.button("Detect Trend"):
    result = detect_trend(symbol, expiry)
    st.success(f"📊 Trend: {result['trend']}")
    st.write("🧠 Reason:", result["reason"])
    st.json(result["supporting_data"])

