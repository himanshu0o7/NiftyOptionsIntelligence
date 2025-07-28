import streamlit as st
import pandas as pd
import time
from datetime import datetime
from SmartApi import SmartConnect
import websocket  # pip install websocket-client
import json
import requests  # For NewsAPI
from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega  # pip install py_vollib for Greeks calculation if needed
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer  # For simple news sentiment; pip install nltk

nltk.download('vader_lexicon', quiet=True)

# --- Page Configuration ---
st.set_page_config(
    page_title="Live Signal Tracker",
    page_icon="🔔",
    layout="wide",
)

st.title("🔔 Live Trading Signal Tracker")

st.markdown("""
This dashboard generates and displays live trading signals for Nifty options using data from Sensibull (via unofficial WebSocket), live news, and executes trades via Angel One SmartAPI.

**Disclaimer:** This is for educational purposes only and is not financial advice. Scraping/WebSocket usage may violate terms; use at own risk.
""")

# --- API Credentials (Store in secrets.toml) ---
try:
    # Angel One
    ANGEL_API_KEY = st.secrets["api_credentials"]["angel_api_key"]
    ANGEL_CLIENT_CODE = st.secrets["api_credentials"]["angel_client_code"]
    ANGEL_PASSWORD = st.secrets["api_credentials"]["angel_password"]
    
    # NewsAPI (free at newsapi.org)
    NEWS_API_KEY = st.secrets["api_credentials"]["news_api_key"]
    
    st.success("API credentials loaded successfully.")
except (FileNotFoundError, KeyError):
    st.error("API credentials not found. Please add them to your `secrets.toml` file. Get NewsAPI key from newsapi.org.")
    st.stop()

# Initialize Angel One SmartConnect
obj = SmartConnect(api_key=ANGEL_API_KEY)
session_data = obj.generateSession(ANGEL_CLIENT_CODE, ANGEL_PASSWORD)
auth_token = session_data['data']['jwtToken']
feed_token = obj.getfeedToken()

# --- Sensibull WebSocket for Data Fetch (Adapted from GitHub repo studiogangster/sensibull-realtime-options-api-ingestor) ---
# Note: This is unofficial; based on public GitHub repo. URL from repo/code inspection.
SENSIBULL_WS_URL = "wss://api.sensibull.com/v1/stream/option_chain?tradingsymbol=NIFTY"  # Example for NIFTY; adjust as per repo

live_data = {}  # To store fetched data: OI, LTP, volume, Greeks, etc.

def on_message(ws, message):
    data = json.loads(message)
    # Parse data: Assuming structure from Sensibull (OI, LTP, volume, Greeks per strike)
    # Example: Update live_data with strikes, calls/puts info
    global live_data
    live_data = data.get('data', {})  # e.g., {'strikes': [...], 'calls': {'OI': ..., 'LTP': ..., 'volume': ..., 'delta': ..., 'gamma': ..., 'theta': ..., 'vega': ...}}

def on_error(ws, error):
    st.error(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_reason):
    st.warning("WebSocket closed. Reconnecting...")
    time.sleep(5)
    connect_sensibull_ws()

def on_open(ws):
    # Subscribe to NIFTY options
    ws.send(json.dumps({"action": "subscribe", "symbols": ["NIFTY"]}))

def connect_sensibull_ws():
    ws = websocket.WebSocketApp(SENSIBULL_WS_URL, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever()

# Start WebSocket in background thread
import threading
ws_thread = threading.Thread(target=connect_sensibull_ws)
ws_thread.daemon = True
ws_thread.start()

# --- Fetch Live News (Using NewsAPI) ---
def fetch_live_news():
    url = f"https://newsapi.org/v2/everything?q=nifty+options+trading&apiKey={NEWS_API_KEY}&pageSize=5&sortBy=publishedAt"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json()['articles']
        return [article['title'] + " " + article['description'] for article in articles]
    return []

# --- Fetch Nifty Spot from Angel One ---
def fetch_live_nifty_spot():
    # Use Angel One API for LTP
    ltp_data = obj.ltpData("NSE", "NIFTY", "26000")  # Symbol token for NIFTY index
    return ltp_data['data']['ltp']

# --- Calculate Greeks if not provided (Fallback) ---
def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    flag = 'c' if option_type == 'call' else 'p'
    d = delta(flag, S, K, T, r, sigma)
    g = gamma(flag, S, K, T, r, sigma)
    t = theta(flag, S, K, T, r, sigma)
    v = vega(flag, S, K, T, r, sigma)
    return {'delta': d, 'gamma': g, 'theta': t, 'vega': v}

# --- Signal Generation Logic (Enhanced with OI, News) ---
def generate_signals(price_history, oi_data, news_texts):
    signals = []
    if len(price_history) < 20:
        return signals

    df = pd.DataFrame(price_history, columns=['price'])
    df['sma_short'] = df['price'].rolling(window=5).mean()
    df['sma_long'] = df['price'].rolling(window=20).mean()

    # OI Analysis (example: high OI change signals buildup)
    oi_change = oi_data.get('oi_change', 0) if oi_data else 0  # From live_data
    if oi_change > 10:  # Threshold
        signals.append({"Reason": "High OI Buildup"})

    # News Sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment_score = np.mean([sia.polarity_scores(text)['compound'] for text in news_texts])
    if sentiment_score > 0.5:
        signals.append({"Reason": "Positive News Sentiment - BUY"})
    elif sentiment_score < -0.5:
        signals.append({"Reason": "Negative News Sentiment - SELL"})

    # MA Crossover
    if df['sma_short'].iloc[-1] > df['sma_long'].iloc[-1] and df['sma_short'].iloc[-2] < df['sma_long'].iloc[-2]:
        signal = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Signal Type": "BUY",
            "Trigger Price": f"{price_history[-1]:.2f}",
            "Reason": "5-period SMA crossed above 20-period SMA + OI/News",
        }
        signals.append(signal)
    elif df['sma_short'].iloc[-1] < df['sma_long'].iloc[-1] and df['sma_short'].iloc[-2] > df['sma_long'].iloc[-2]:
        signal = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Signal Type": "SELL",
            "Trigger Price": f"{price_history[-1]:.2f}",
            "Reason": "5-period SMA crossed below 20-period SMA + OI/News",
        }
        signals.append(signal)
    
    return signals

# --- Streamlit App Layout ---
if 'price_history' not in st.session_state:
    st.session_state.price_history = []
if 'signals_log' not in st.session_state:
    st.session_state.signals_log = pd.DataFrame(columns=["Timestamp", "Signal Type", "Trigger Price", "Reason"])

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.subheader("Market Status")
    nifty_spot_placeholder = st.empty()
    oi_placeholder = st.empty()
    greeks_placeholder = st.empty()

with col2:
    st.subheader("Price Chart")
    chart_placeholder = st.empty()

with col3:
    st.subheader("Live News")
    news_placeholder = st.empty()

st.subheader("Signal Log")
signals_log_placeholder = st.empty()

# --- Auto-Refreshing Loop ---
while True:
    # 1. Fetch Data
    current_nifty_spot = fetch_live_nifty_spot()
    st.session_state.price_history.append(current_nifty_spot)
    if len(st.session_state.price_history) > 100:
        st.session_state.price_history.pop(0)
    
    # Sensibull data (OI, LTP, volume, Greeks from live_data)
    oi_data = live_data.get('oi', {})  # Example parsing
    ltp = live_data.get('ltp', 0)
    volume = live_data.get('volume', 0)
    greeks = live_data.get('greeks', {}) or calculate_greeks(current_nifty_spot, 25000, 0.01, 0.05, 0.2)  # Fallback

    # News
    news_texts = fetch_live_news()

    # 2. Generate Signals
    new_signals = generate_signals(st.session_state.price_history, oi_data, news_texts)
    if new_signals:
        st.session_state.signals_log = pd.concat([st.session_state.signals_log, pd.DataFrame(new_signals)], ignore_index=True)
        
        # Execute Trade via Angel One on Signal
        for signal in new_signals:
            if signal['Signal Type'] == 'BUY':
                order_params = {
                    "variety": "NORMAL",
                    "tradingsymbol": "NIFTY25JUL25000CE",  # Example; dynamic based on signal
                    "symboltoken": "12345",  # Fetch token
                    "transactiontype": "BUY",
                    "exchange": "NFO",
                    "ordertype": "MARKET",
                    "producttype": "INTRADAY",
                    "duration": "DAY",
                    "price": "0",
                    "squareoff": "0",
                    "stoploss": "0",
                    "quantity": 50  # Lot size
                }
                order_id = obj.placeOrder(order_params)
                st.success(f"BUY Order Executed: {order_id}")
            # Similar for SELL

    # 3. Update UI
    with col1:
        nifty_spot_placeholder.metric("Nifty Spot", f"{current_nifty_spot:.2f}")
        oi_placeholder.metric("Open Interest", oi_data.get('total', 0))
        greeks_placeholder.json(greeks)  # Display Greeks

    with col2:
        price_df = pd.DataFrame({"Nifty Spot": st.session_state.price_history})
        chart_placeholder.line_chart(price_df)

    with col3:
        news_placeholder.write("\n".join(news_texts))

    signals_log_placeholder.dataframe(st.session_state.signals_log, use_container_width=True)

    time.sleep(10)

