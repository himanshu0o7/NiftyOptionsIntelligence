"""
Streamlit dashboard for the Nifty Options Intelligence bot.

This script stitches together the live option stream UI, token refresh
management and a simple trend–detector page.  Tokens are refreshed
periodically via `SessionManager`, and WebSocket feeds can be spawned
in the background for continuous ticks.  A separate section allows
users to run the trend detection logic on demand.

Environment variables required for this app to function should be
provided in a `.env` file or in your shell:

* ``ANGEL_API_KEY`` – your Angel One API key
* ``ANGEL_CLIENT_ID`` – your Angel One client/user ID
* ``ANGEL_PIN`` – your trading PIN/password
* ``ANGEL_TOTP_SECRET`` – the base32 secret used for generating OTPs

If any of these are missing, ``SessionManager`` will raise an
appropriate error.
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import time
import subprocess
import signal

from session_manager import SessionManager
from option_stream_ui import get_option_data
from utils.trend_detector import detect_trend


# ---------------------------------------------------------------------------
# Streamlit page setup
# ---------------------------------------------------------------------------
# Use a wide layout and automatically refresh the page every 5 seconds to
# display fresh LTP data.  The `limit` keeps the auto‑refresh from running
# indefinitely during development.
st.set_page_config(layout="wide")
st_autorefresh(interval=5000, limit=100, key="refresh_autokey_001")
st.title("📈 Nifty Options Intelligence Dashboard")
st.markdown("Use this app to monitor live CE/PE option data for NIFTY/BANKNIFTY and assess short‑term trends.")


# ---------------------------------------------------------------------------
# Token refresh support
# ---------------------------------------------------------------------------
# The Angel One tokens have a limited lifetime; refresh them every 14 minutes.
tokens = None
last_login_time = 0


def ensure_tokens_fresh() -> None:
    """Refresh the Angel One session tokens if they have expired.

    A cached session is reused until 14 minutes have elapsed.  On
    expiration, a new session is created via ``SessionManager`` and
    stored in the global ``tokens`` variable.
    """
    global tokens, last_login_time
    # 14 minutes (14 × 60 seconds)
    refresh_interval = 14 * 60
    if tokens is None or time.time() - last_login_time > refresh_interval:
        sm = SessionManager()
        session = sm.get_session()
        tokens = session
        last_login_time = time.time()


# ---------------------------------------------------------------------------
# Option data section
# ---------------------------------------------------------------------------
def option_data_ui() -> None:
    """Render the option LTP viewer.

    Users select a symbol (e.g. NIFTY or BANKNIFTY), strike and option type.
    The latest quote is displayed in a table.  A button spawns the
    ``websocket_runner.py`` script in a background subprocess so that
    real‑time ticks continue streaming.
    """
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Select Symbol", ["NIFTY", "BANKNIFTY"])
        option_type = st.radio("Option Type", ["CE", "PE"], horizontal=True)
    with col2:
        strike_price = st.number_input(
            "Select Strike Price",
            min_value=10000,
            max_value=50000,
            step=50,
            value=22500,
        )

    try:
        ensure_tokens_fresh()
        # Delegated fetch from option_stream_ui; can be replaced with live logic.
        data = get_option_data(symbol, strike_price, option_type)
        if data and "error" not in data:
            st.subheader(f"📊 Live Data for {symbol} {strike_price} {option_type}")
            st.dataframe(pd.DataFrame([data]))
        else:
            st.error(data.get("error", "Unknown error occurred"))
    except Exception as e:
        st.error(f"⚠️ App error: {e}")

    # Control to start a persistent WebSocket feed in a background process.
    if st.button("Start Live WebSocket"):
        # Launch the websocket runner using subprocess so it does not block
        # the Streamlit thread.  In production you might integrate the
        # WebSocket logic directly instead of spawning a new Python process.
        subprocess.Popen(["python3", "websocket_runner.py"])
        st.success("WebSocket started in background.")


# ---------------------------------------------------------------------------
# Trend detector section
# ---------------------------------------------------------------------------
def trend_detector_ui() -> None:
    """Render a simple form to detect the market trend using delta and OI change."""
    st.markdown("---")
    st.subheader("📈 Market Trend Detector")
    # Use distinct keys for Streamlit widgets to avoid collisions with the
    # option section above.
    symbol = st.selectbox("Choose Symbol", ["NIFTY", "BANKNIFTY"], key="trend_symbol")
    expiry = st.text_input(
        "Enter Expiry (e.g., 25JUL2025)",
        key="trend_expiry",
        placeholder="25JUL2025",
    )
    if st.button("Detect Trend"):
        result = detect_trend(symbol, expiry)
        if result.get("trend") == "Error":
            st.error(result["reason"])
        else:
            st.success(f"📊 Trend: {result['trend']}")
            st.write("🧠 Reason:", result["reason"])
            st.json(result["supporting_data"])


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for Streamlit.

    Compose the UI by rendering the option data viewer and the trend
    detector on the same page.  A dummy signal handler is installed
    when running in the main thread to gracefully handle SIGTERM
    during local development.
    """
    option_data_ui()
    trend_detector_ui()
    # Optional signal handling for graceful shutdown
    try:
        # Only register the handler if running in the main thread
        if signal.getsignal(signal.SIGTERM) != signal.SIG_DFL:
            def handler(signum, frame):
                print(f"Signal {signum} received")
            signal.signal(signal.SIGTERM, handler)
    except Exception:
        pass


if __name__ == "__main__":
    main()