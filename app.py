# app.py - Fixed and cleaned version with safe threading and no duplicate key issue

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import time
import threading

# Local module imports (make sure these files exist and are error-free)
from session_manager import SessionManager
from option_stream_ui import fetch_option_data

# ---------------------------
# PAGE SETUP
# ---------------------------
st.set_page_config(layout="wide")
# Commenting out auto-refresh for better table visibility
# st_autorefresh(interval=5000, limit=100, key="refresh_autokey_001")

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
    if time.time() - last_login_time > (14 * 60):  # refresh every 14 min
        try:
            time.sleep(1)  # prevent aggressive retry
            sm = SessionManager()
            session = sm.get_session()
            tokens = session
            last_login_time = time.time()
        except Exception as e:
            # Handle missing credentials gracefully for demo
            st.warning(f"Running in demo mode: {str(e)}")
            tokens = {"demo": True}
            last_login_time = time.time()

# ---------------------------
# STREAMLIT UI
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])
    option_type = st.radio("Option Type", ["CE", "PE"], horizontal=True)

with col2:
    strike_price = st.number_input(
        "Select Strike Price", min_value=10000, max_value=50000, step=50, value=22500
    )

# ---------------------------
# DATA FETCH + DISPLAY
# ---------------------------
try:
    ensure_tokens_fresh()
    data = fetch_option_data(symbol, strike_price, option_type)

    if data and "error" not in data:
        st.subheader(f"📊 Live Data for {symbol} {strike_price} {option_type}")
        
        # Create a comprehensive DataFrame with all important metrics
        display_data = {
            "Metric": ["Symbol", "LTP", "Bid", "Ask", "Volume", "OI", "OI Change", 
                      "Delta", "Gamma", "Theta", "Vega", "IV", "Change", "Change %", "Last Updated"],
            "Value": [
                data.get("symbol", "N/A"),
                f"₹{data.get('ltp', 0):.2f}",
                f"₹{data.get('bid', 0):.2f}",
                f"₹{data.get('ask', 0):.2f}",
                f"{data.get('volume', 0):,}",
                f"{data.get('oi', 0):,}",
                f"{data.get('oi_change', 0):+,}",
                f"{data.get('delta', 0):.4f}",
                f"{data.get('gamma', 0):.6f}",
                f"{data.get('theta', 0):.4f}",
                f"{data.get('vega', 0):.4f}",
                f"{data.get('iv', 0):.2f}%",
                f"₹{data.get('change', 0):+.2f}",
                f"{data.get('change_percent', 0):+.2f}%",
                data.get("last_update", "N/A")
            ]
        }
        
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True)
        
        # Additional summary metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("LTP", f"₹{data.get('ltp', 0):.2f}", f"{data.get('change_percent', 0):+.2f}%")
        with col2:
            st.metric("Volume", f"{data.get('volume', 0):,}")
        with col3:
            st.metric("Open Interest", f"{data.get('oi', 0):,}", f"{data.get('oi_change', 0):+,}")
        with col4:
            st.metric("IV", f"{data.get('iv', 0):.2f}%")
            
    else:
        st.error(data.get("error", "Unknown error occurred"))

except Exception as e:
    st.error(f"⚠️ App error: {e}")

# ---------------------------
# SAFELY HANDLE SIGNALS IF NEEDED
# ---------------------------
try:
    import signal
    if threading.current_thread() is threading.main_thread():
        def handler(signum, frame):
            print(f"Signal {signum} received")
        signal.signal(signal.SIGTERM, handler)
except Exception as err:
    print(f"[Signal Handling Skipped] Reason: {err}")

import subprocess

if st.button("Start Live WebSocket"):
    subprocess.Popen(["python3", "websocket_runner.py"])
    st.success("WebSocket started in background.")

