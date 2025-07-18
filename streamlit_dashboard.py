# streamlit_dashboard.py - 📊 Real-time LTP Chart Dashboard

import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import json
import time
import os

st.set_page_config(page_title="Options LTP Live Chart", layout="wide")
st.title("📊 Live Option LTP Chart - Angel One WebSocket Feed")

# ---- Settings ----
FEED_FILE = "ltp_feed.json"
REFRESH_INTERVAL = 5  # seconds

def load_feed_data():
    if not os.path.exists(FEED_FILE):
        return pd.DataFrame(columns=['symbol', 'ltp', 'timestamp'])

    with open(FEED_FILE, 'r') as f:
        raw_data = json.load(f)

    df = pd.DataFrame(raw_data)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def plot_ltp_chart(symbol_df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=symbol_df['timestamp'],
        y=symbol_df['ltp'],
        mode='lines+markers',
        name=symbol,
        line=dict(width=2)
    ))

    fig.update_layout(
        title=f"Live LTP for {symbol}",
        xaxis_title="Time",
        yaxis_title="LTP",
        template="plotly_dark",
        height=500
    )
    return fig

# ---- Main App ----
data = load_feed_data()
if data.empty:
    st.warning("Waiting for live data from WebSocket...")
    st.stop()

symbols = sorted(data['symbol'].unique())
selected_symbol = st.selectbox("Select Symbol to Plot", symbols)

symbol_data = data[data['symbol'] == selected_symbol].sort_values('timestamp')
chart = plot_ltp_chart(symbol_data, selected_symbol)
st.plotly_chart(chart, use_container_width=True)

# Optional: Auto-refresh
st.markdown("---")
st.info(f"Auto-refreshes every {REFRESH_INTERVAL} seconds")
time.sleep(REFRESH_INTERVAL)
st.experimental_rerun()

