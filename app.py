# fix-bot-2025-07-24
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
import logging
import sys

# Import production configuration and error handling
try:
    from production_config import get_config, rate_limited, angel_api_limiter, with_retry
    config = get_config()  # Only create when needed
    logger = logging.getLogger(__name__)
    logger.info("✅ Production configuration loaded")
except ImportError as e:
    st.error(f"❌ Failed to load production configuration: {e}")
    st.stop()
except ValueError as e:
    st.error(f"❌ Configuration error: {e}")
    st.error("Please check your environment variables in .env file")
    st.stop()

try:
    from session_manager import SessionManager
    from option_stream_ui import get_option_data
except ImportError as e:
    st.error(f"❌ Failed to import required modules: {e}")
    st.error("Please ensure all dependencies are installed and modules are available.")
    st.stop()
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

websocket_process = None  # Global variable for WebSocket process management

@with_retry(max_retries=3, delay=2)
@rate_limited(angel_api_limiter)
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
        try:
            logger.info("🔄 Refreshing Angel One session tokens...")
            session_mgr = SessionManager()
            tokens = session_mgr.get_session()
            last_login_time = time.time()
            logger.info("✅ Session refreshed successfully")
            st.success("🔐 Session refreshed successfully")
        except Exception as exc:
            error_msg = f"❌ Session refresh failed: {exc}"
            logger.error(error_msg)
            st.error(error_msg)
            tokens = None
            raise

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
# TOKEN REFRESH FUNCTION (redundant - using the decorated version above)
# ---------------------------

# fix-bot-2025-07-24

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

    # Control to start and stop a persistent WebSocket feed in a background process.
    global websocket_process
    if st.button("Start Live WebSocket"):
        # Launch the websocket runner using subprocess so it does not block
        # the Streamlit thread.  In production you might integrate the
        # WebSocket logic directly instead of spawning a new Python process.
        if websocket_process is None or websocket_process.poll() is not None:
            try:
                websocket_process = subprocess.Popen(["python3", "websocket_runner.py"])
                st.success("WebSocket started in background.")
            except Exception as e:
                st.error(f"Failed to start WebSocket: {e}")
        else:
            st.warning("WebSocket is already running.")

    if st.button("Stop Live WebSocket"):
        if websocket_process is not None and websocket_process.poll() is None:
            websocket_process.terminate()
            websocket_process.wait()
            st.success("WebSocket stopped.")
        else:
            st.warning("No WebSocket process is running.")
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
