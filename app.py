import streamlit as st
import pandas as pd
import logging
import time

from session_manager import SessionManager
from option_stream_ui import run_dashboard

run_dashboard()


# Set up the page
st.set_page_config(page_title="Nifty Options Intelligence", layout="wide")
st.title("🔍 Nifty Options Intelligence Dashboard")
st.markdown("""
This dashboard shows real-time options data with Greeks, OI, volume, and signal flags using Sensibull + Angel One API.
""")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    refresh_rate = st.slider("Refresh Interval (seconds)", 5, 60, 15)

# Placeholder for data
placeholder = st.empty()

# Streamlit-safe loop (no threads)
while True:
    with placeholder.container():
        try:
            df = get_option_data()
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error fetching option data: {e}")
        st.toast("📱 Data updated")

    time.sleep(refresh_rate)
    # Optional: Add break condition if needed for development/testing
    # For example: if st.button("Stop"): break

