# option_stream_ui.py - Streamlit chart UI for live LTP and signals

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from threading import Thread
from collections import deque
from telegram_alerts import send_telegram_alert
import numpy as np

def fetch_option_data(symbol, strike_price, option_type):
    """
    Fetch live option data including LTP, OI, Greeks
    Returns structured data compatible with the dashboard
    """
    try:
        # Generate sample data for demo purposes
        # In production, this would connect to actual APIs
        
        # Basic option data structure
        option_data = {
            "symbol": f"{symbol}{strike_price}{option_type}",
            "strike": strike_price,
            "option_type": option_type,
            "underlying": symbol,
            "expiry": "10JUL2025",
            
            # Live market data
            "ltp": round(np.random.uniform(50, 300), 2),
            "bid": round(np.random.uniform(45, 295), 2),
            "ask": round(np.random.uniform(55, 305), 2),
            "volume": np.random.randint(1000, 50000),
            "oi": np.random.randint(50000, 500000),
            "oi_change": np.random.randint(-10000, 10000),
            
            # Greeks
            "delta": round(np.random.uniform(0.1, 0.9) if option_type == "CE" else np.random.uniform(-0.9, -0.1), 4),
            "gamma": round(np.random.uniform(0.001, 0.01), 6),
            "theta": round(np.random.uniform(-2, -0.1), 4),
            "vega": round(np.random.uniform(0.1, 2), 4),
            "iv": round(np.random.uniform(15, 35), 2),
            
            # Additional metrics
            "change": round(np.random.uniform(-20, 20), 2),
            "change_percent": round(np.random.uniform(-15, 15), 2),
            "last_update": datetime.now().strftime("%H:%M:%S"),
        }
        
        return option_data
        
    except Exception as e:
        return {"error": f"Failed to fetch option data: {str(e)}"}

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

