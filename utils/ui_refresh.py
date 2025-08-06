import streamlit as st
import time
import os
from datetime import datetime
from telegram_handler import send_alert  # ensure this module is set up

LOG_FILE = "logs/refresh_history.log"
os.makedirs("logs", exist_ok=True)

def streamlit_autorefresh(seconds: int = 30, enable_telegram: bool = False, enable_debug_panel: bool = False):
    """Auto-refresh Streamlit app every `seconds` seconds with countdown, alert, pause toggle, and logging."""

    # Sidebar: Controls
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        refresh_toggle = st.checkbox("⏸️ Pause Auto Refresh", value=False)
    with col2:
        debug_toggle = st.checkbox("🧪 Show Debug", value=enable_debug_panel)

    # Exit if paused
    if refresh_toggle:
        st.markdown("🔁 Auto-refresh is currently **paused**")
        return

    # Countdown display
    placeholder = st.empty()
    countdown = seconds

    for sec in range(countdown, 0, -1):
        with placeholder.container():
            st.markdown(
                f"<div style='text-align:center; font-size:18px;'>⏳ Refreshing in `{sec}` seconds...</div>",
                unsafe_allow_html=True
            )
        time.sleep(1)

    # On refresh event
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    msg = f"🔄 Refreshed at {ts}"

    # Log it
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

    if enable_telegram:
        send_alert(msg)

    if debug_toggle:
        with st.sidebar.expander("🧪 Refresh Debug Log"):
            with open(LOG_FILE, 'r') as f:
                logs = f.read().splitlines()[-20:]
                for log in logs:
                    st.code(log, language='log')

    st.rerun()
