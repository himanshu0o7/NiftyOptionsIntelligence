# option_stream_ui.py - Streamlit chart UI for live LTP and signals

import streamlit as st
import pandas as pd
import time
import requests
from datetime import datetime
from threading import Thread
from collections import deque
from telegram_alerts import send_telegram_alert

def fetch_option_data(symbol: str, expiry_date: str) -> pd.DataFrame:
    """
    Fetch option chain and Greeks for a given symbol and expiry date.
    Returns a DataFrame with LTP, OI, and Greeks (Delta, Gamma, Theta, Vega).
    """
    # Example endpoint (replace with your actual data source)
    url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()["records"]["data"]
    
    records = []
    for entry in data:
        for ce_pe in ["CE", "PE"]:
            if ce_pe in entry and entry[ce_pe].get("expiryDate") == expiry_date:
                option = entry[ce_pe]
                record = {
                    "strikePrice": option["strikePrice"],
                    "type": ce_pe,
                    "LTP": option.get("lastPrice"),
                    "OI": option.get("openInterest"),
                    "changeOI": option.get("changeinOpenInterest"),
                    "Delta": option.get("delta"),
                    "Gamma": option.get("gamma"),
                    "Theta": option.get("theta"),
                    "Vega": option.get("vega"),
                }
                records.append(record)
    df = pd.DataFrame(records)
    return df

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

