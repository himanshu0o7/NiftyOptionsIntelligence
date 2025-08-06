import logging
from datetime import datetime

import pandas as pd
import plotly.express as px
import pytz
import streamlit as st

from greek_scanner_panel import render_scanner
from option_stream_ui import get_nifty_option_tokens
from telegram_handler import send_alert
from utils.sensibull_greeks_fetcher import fetch_option_data
from utils.ui_refresh import streamlit_autorefresh

# Somewhere below main content
st.divider()
render_scanner()


# 📄 Streamlit Page Config
st.set_page_config(layout="wide", page_title="📈 Nifty Live Option Dashboard")

# 🔧 Logger Setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 🔁 Auto-Refresh Controls
st.sidebar.markdown("## 🔄 Auto Refresh Settings")
refresh_interval = st.sidebar.selectbox(
    "Refresh Interval (seconds)", [15, 30, 60], index=1
)
pause_refresh = st.sidebar.checkbox("⏸️ Pause Auto-Refresh", value=False)

if not pause_refresh:
    streamlit_autorefresh(
        seconds=refresh_interval,
        enable_telegram=True,
        enable_debug_panel=True
    )
    st.sidebar.success(f"🔁 Refreshing every {refresh_interval} sec")
else:
    st.sidebar.warning("⏸️ Auto-refresh is paused")

# 🕒 Market Hours
IST = pytz.timezone("Asia/Kolkata")


def is_market_open():
    """Check if the current time is within Indian market hours."""
    now = datetime.now(IST)
    return (
        now.replace(hour=9, minute=15)
        <= now
        <= now.replace(hour=15, minute=30)
    )


# 🧠 Load Token Data
ce_df, pe_df = get_nifty_option_tokens()
if ce_df is None or ce_df.empty:
    st.error("⚠️ Could not load Nifty option tokens.")
    st.stop()

# 🎛️ User Filters
strikes = sorted(ce_df["strike"].unique())
expiries = sorted(ce_df["expiry"].unique())

selected_strike = st.sidebar.selectbox(
    "🎯 Select Strike", strikes, index=len(strikes) // 2
)
selected_expiry = st.sidebar.selectbox("🗓️ Select Expiry", expiries)

token_df = pd.concat(
    [
        ce_df[
            (ce_df["strike"] == selected_strike)
            & (ce_df["expiry"] == selected_expiry)
        ],
        pe_df[
            (pe_df["strike"] == selected_strike)
            & (pe_df["expiry"] == selected_expiry)
        ],
    ]
)

# 📊 Main Dashboard
try:
    if not is_market_open():
        st.warning(
            "⚠️ Market is closed (9:15 AM – 3:30 PM IST). Data may be stale."
        )

    df = fetch_option_data(token_df['token'].tolist())

    if df.empty:
        st.error("🚫 No option data found. Check filters or API connectivity.")
        st.stop()

    st.success("✅ Live Option Data Fetched")
    st.write(f"🧮 Data Shape: {df.shape}")

    # 📊 OI + Volume Chart
    fig = px.bar(df, x='Token', y=['oi', 'volume'], color='type',
                 title="OI vs Volume", barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    # 📉 Greeks Chart
    if all(col in df.columns for col in ['delta', 'theta', 'vega']):
        greek_chart = px.line(df, x='strike', y=['delta', 'theta', 'vega'],
                              title="Option Greeks by Strike")
        st.plotly_chart(greek_chart, use_container_width=True)

    # 🔎 Full Data Table
    st.dataframe(df)

except Exception as e:
    logger.exception("❌ Fetch failed")
    st.error(f"❌ Error: {e}")
    try:
        send_alert(f"❌ Fetch failed in app.py: {e}")
    except Exception as alert_err:
        logger.error(f"Telegram alert failed: {alert_err}")
