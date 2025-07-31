import requests
import pytz
import streamlit as st
import pandas as pd
import logging
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from session_manager import SessionManager
from option_stream_ui import get_option_data
from datetime import datetime


logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.ERROR)
logging.basicConfig(level=logging.DEBUG)  # Debug for detailed logs
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Nifty Options Intelligence", layout="wide")
st.title("🔍 Nifty Options Intelligence Dashboard")

st.markdown("""
This dashboard shows real-time options data with Greeks, OI, volume, and signal flags using Sensibull + Angel One API.
""")

with st.sidebar:
    st.header("⚙️ Settings")
    refresh_rate = st.slider("Refresh Interval (seconds)", 5, 60, 15)

# Auto-refresh every refresh_rate seconds (in ms)
st_autorefresh(interval=refresh_rate * 1000, key="datarefresh")

# Check market hours
import pytz

def is_market_open():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close

# Fetch and render data
try:
    if not is_market_open():
        st.warning("Market is closed (9:15 AM - 3:30 PM IST, Mon-Fri). Showing cached or no data.")
    df = get_option_data()
    if df.empty:
        st.error("No option data available. Check market hours or API settings.")
        logger.warning("Empty DataFrame from get_option_data")
    else:
        logger.debug(f"DataFrame shape: {df.shape}")
        st.write(f"Data shape: {df.shape}")  # Debug UI output
        fig = px.bar(df, x='Token', y=['oi', 'volume'], color='type' if 'type' in df else None,
                     title="Live OI and Volume by Option Token", barmode='group')
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        if all(col in df for col in ['delta', 'theta', 'vega']):
            greeks_fig = px.line(df, x='strike' if 'strike' in df else 'Token', y=['delta', 'theta', 'vega'],
                                 title="Greeks by Strike/Token")
            st.plotly_chart(greeks_fig, use_container_width=True, theme="streamlit")
        st.dataframe(df, use_container_width=True)
        st.toast("📡 Data updated")
except IndexError as e:
    st.error(f"Data indexing error: {e}. Check filters or scrip master data.")
    logger.error(f"IndexError in data fetch: {e}", exc_info=True)
except ValueError as e:
    st.error(f"Data validation error: {e}. Check API response or token filters.")
    logger.error(f"ValueError in data fetch: {e}", exc_info=True)
except requests.exceptions.RequestException as e:
    st.error(f"API connection error: {e}. Check internet or API credentials.")
    logger.error(f"RequestException in data fetch: {e}", exc_info=True)
except Exception as e:
    st.error(f"Unexpected error: {e}. See logs for details.")
    logger.error(f"Unexpected error: {e}", exc_info=True)

