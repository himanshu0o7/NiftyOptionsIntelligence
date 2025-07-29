import streamlit as st
import pandas as pd
import logging
import threading
import time

from session_manager import SessionManager
from option_stream_ui import get_option_data

st.set_page_config(page_title="Nifty Options Intelligence", layout="wide")
st.title("🔍 Nifty Options Intelligence Dashboard")

st.markdown("""
This dashboard shows real-time options data with Greeks, OI, volume, and signal flags using Sensibull + Angel One API.
""")

with st.sidebar:
    st.header("⚙️ Settings")
    refresh_rate = st.slider("Refresh Interval (seconds)", 5, 60, 15)

placeholder = st.empty()

def stream_data():
    while True:
        with placeholder.container():
            try:
                df = get_option_data()
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Error fetching option data: {e}")
            st.toast("📡 Data updated")
        time.sleep(refresh_rate)

thread = threading.Thread(target=stream_data)
thread.daemon = True
thread.start()