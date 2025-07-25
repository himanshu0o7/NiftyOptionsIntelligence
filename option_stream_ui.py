# option_stream_ui.py - Streamlit chart UI for live LTP and signals

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from threading import Thread
from collections import deque
from telegram_alerts import send_telegram_alert
from utils.option_data_utils import fetch_option_data


# Shared memory store (can also use cache or session state)
live_data_store = {
    'ltp': {},
    'triggered': set()
}

st.set_page_config(layout="wide")
st.title("📈 Live Option Stream Monitor")

col1, col2 = st.columns(2)

with col1:
    selected_symbols = st.multiselect("Select Option Symbols (format: SYMBOL|TOKEN)", [
        "NIFTY25JUL25000CE|123456",
        "BANKNIFTY25JUL45000PE|654321"
    ])
    upper_threshold = st.number_input("Upper Price Alert Threshold", min_value=0.0, value=100.0)
    lower_threshold = st.number_input("Lower Price Alert Threshold", min_value=0.0, value=50.0)

with col2:
    auto_trade = st.checkbox("Auto Place Order on Trigger")
    start_stream = st.button("▶️ Start Stream")

ltp_chart = st.empty()

# Append live LTP data to deque
ltp_history = {sym.split("|")[0]: deque(maxlen=100) for sym in selected_symbols}

# Live UI update loop
def update_ui():
    while True:
        display_df = []
        for entry in selected_symbols:
            sym, token = entry.split("|")
            ltp = live_data_store['ltp'].get(token, None)
            if ltp:
                timestamp = datetime.now().strftime("%H:%M:%S")
                ltp_history[sym].append((timestamp, ltp))
                display_df.append({"Symbol": sym, "LTP": ltp, "Time": timestamp})

                # Trigger alerts
                if token not in live_data_store['triggered']:
                    if ltp > upper_threshold:
                        send_telegram_alert(f"📈 {sym} crossed upper threshold! LTP: ₹{ltp}")
                        live_data_store['triggered'].add(token)
                    elif ltp < lower_threshold:
                        send_telegram_alert(f"📉 {sym} fell below lower threshold! LTP: ₹{ltp}")
                        live_data_store['triggered'].add(token)
                    # Optional: place_order(symbol, token, "SELL")

        if display_df:
            df = pd.DataFrame(display_df)
            ltp_chart.table(df)

        time.sleep(2)

if start_stream:
    Thread(target=update_ui, daemon=True).start()
    st.success("🟢 Stream started. Waiting for LTP data...")

# Add this at the bottom or top of option_stream_ui.py

def get_option_data(symbol, strike_price, option_type):
    """
    Dummy or real implementation to return option LTP data.
    You can replace this with actual logic to fetch data.
    """
    option_symbol = f"{symbol}25JUL{strike_price}{option_type}"
    return {
        "symbol": option_symbol,
        "ltp": 150.25,  # Replace with live LTP if available
        "time": time.strftime("%H:%M:%S")
    }

"""
Real-time CE/PE screener Streamlit UI with live Sensibull Greeks integration.
"""

import streamlit as st
from utils.sensibull_greeks_fetcher import fetch_option_data


def show_option_stream():
    st.title("📡 Live Option Stream (with Sensibull Greeks)")

    index = st.selectbox("Select Index", ["NIFTY", "BANKNIFTY"])
    strike = st.number_input("Select Strike", value=25200)
    option_type = st.radio("Option Type", ["CE", "PE"])

    with st.spinner("Fetching live data from Sensibull..."):
        option_data = fetch_option_data(index, strike, option_type)

    if option_data:
        col1, col2, col3 = st.columns(3)
        col1.metric("Delta", f"{option_data['delta']:.2f}")
        col2.metric("LTP", f"₹{option_data['ltp']:.2f}")
        col3.metric("IV", f"{option_data['iv']}%")

        if 0.4 <= option_data["delta"] <= 0.7:
            st.success("✅ Signal Confirmed: Delta in ideal zone for directional trade")
        else:
            st.warning("⚠️ Delta not in optimal range")
    else:
        st.error("No matching option data found or API failed.")


# Optional wrapper
if __name__ == "__main__":
    show_option_stream()

"""
Module for fetching option stream data from Angel One API.
This provides utility functions to fetch real-time option data.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_option_data(symbol: str, strike_price: int, option_type: str) -> Dict[str, Any]:
    """
    Fetch option data for the given symbol, strike price and option type.

    Args:
        symbol (str): The symbol (e.g., "NIFTY", "BANKNIFTY")
        strike_price (int): The strike price
        option_type (str): Option type ("CE" or "PE")

    Returns:
        dict: Option data including LTP, change, OI, etc.
    """
    try:
        # This is a placeholder implementation - in production, replace with actual API call
        # In a real implementation, you would use the Angel One API client to fetch live data

        # Mock data for demonstration
        return {
            "symbol": f"{symbol} {strike_price} {option_type}",
            "ltp": 250.45,
            "change": 2.34,
            "change_percent": 0.94,
            "open_interest": 1450,
            "volume": 2345,
            "bid": 250.30,
            "ask": 250.60,
            "timestamp": "2025-07-25 02:15:10"
        }
    except Exception as e:
        logger.error(f"Error fetching option data: {e}")
        return {"error": f"Failed to fetch option data: {str(e)}"}
"""
Option stream UI logic - stub for get_option_data.

Replace this logic with actual option chain/Greeks fetching.
"""

def get_option_data(symbol: str, strike: int, option_type: str):
    # Dummy data for UI test
    return {
        "symbol": symbol,
        "strike": strike,
        "option_type": option_type,
        "price": 123.45,
        "oi": 1000,
        "change": 0.5,
        "timestamp": "2025-07-25 14:10:00"
    }


