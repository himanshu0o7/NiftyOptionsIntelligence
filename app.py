import time
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ANGEL_API_KEY") or "your_api_key"
CLIENT_ID = os.getenv("ANGEL_CLIENT_ID") or "H60779"
MPIN = os.getenv("ANGEL_PIN") or "1234"
TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET") or "xxxxxxxxxxxxxxx"

if not all([API_KEY, CLIENT_ID, MPIN, TOTP_SECRET]):
    st.error("❌ API credentials not loaded properly. Please check .env.")
    st.stop()


# Load .env explicitly from root directory
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

# Safely import streamlit
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 10 sec (10000 ms)
st_autorefresh(interval=10000, limit=None, key="greek_refresh")

try:
    import streamlit as st
except ModuleNotFoundError:
    print("❌ Streamlit not installed. Please run: pip install streamlit")
else:
    # Import custom modules
    try:
        from login_manager import AngelOneLogin
        from greeks_handler import fetch_option_greeks
        from order_executor import place_order
        from utils.oi_data import fetch_oi_buildup
        from utils.historical import get_historical_candles
        from utils.websockets import start_websocket_feed
    except ImportError as e:
        st.error(f"❌ Module Import Error: {e}")
        st.stop()

    # Validate .env credentials
    API_KEY = os.getenv("ANGEL_API_KEY")
    CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
    TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET")
    MPIN = os.getenv("ANGEL_PIN")

    if not all([API_KEY, CLIENT_ID, TOTP_SECRET, MPIN]):
        st.error("❌ API credentials not loaded. Check .env file location or contents.")
        st.stop()

    # UI Title
    st.title("🔁 Nifty Options Intelligence | Live Trading Dashboard")

    # Login Block
    try:
        angel = AngelOneLogin(API_KEY, CLIENT_ID, MPIN, TOTP_SECRET)
        tokens = angel.login(force_refresh=True)
        st.success(f"✅ Login successful for {tokens.get('clientcode', 'unknown')}")
    except Exception as e:
        st.error(f"❌ Login failed: {e}")
        st.stop()

    # Symbol selection
    symbol = st.selectbox("📈 Choose Symbol", ["NIFTY", "BANKNIFTY"])
    strike = st.number_input("📌 Enter Strike Price", value=23500)
    option_type = st.radio("📄 Option Type", ["CE", "PE"])

    # Market Data Fetch
    if st.button("🛱️ Fetch Market Data"):
        try:
            greeks = fetch_option_greeks(symbol, strike, option_type, tokens)
            oi_data = fetch_oi_buildup(symbol)
            st.subheader("🧠 Option Greeks")
            st.json(greeks)
            st.subheader("📊 OI Buildup")
            st.json(oi_data)
        except Exception as e:
            st.error(f"Error fetching live data: {e}")

    # Historical Candles
    with st.expander("📜 Show Historical Data"):
        try:
            df = get_historical_candles(symbol)
            st.dataframe(df.tail(20))
        except Exception as e:
            st.error(f"Historical Data Error: {e}")

    # WebSocket Feed
    if st.button("🔌 Start WebSocket Feed"):
        try:
            start_websocket_feed(tokens)
            st.success("WebSocket feed started.")
        except Exception as e:
            st.error(f"WebSocket Error: {e}")

    # Live Order Form
    with st.form("📥 Execute Live Order"):
        qty = st.number_input("Number of Lots", min_value=1, value=1)
        product = st.selectbox("Product", ["MIS", "NRML"])
        transaction_type = st.radio("Transaction Type", ["BUY", "SELL"])
        submitted = st.form_submit_button("🚀 Place Order")
        if submitted:
            try:
                resp = place_order(symbol, strike, option_type, qty, product, transaction_type, tokens)
                st.success(f"✅ Order placed: {resp}")
            except Exception as e:
                st.error(f"Order Error: {e}")

# Automatic Greek + OI Fetch
try:
    greeks = fetch_option_greeks(symbol, strike, option_type, tokens)
    oi_data = fetch_oi_buildup(symbol)

    st.subheader("🧠 Option Greeks (updated every 10 sec)")
    st.json(greeks)
    st.subheader("📊 OI Buildup")
    st.json(oi_data)

except Exception as e:
    st.error(f"Error fetching live data: {e}")

import sqlite3
import requests
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# Telegram config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Auto-refresh every 10s
st_autorefresh(interval=10000, limit=None, key="refresh_greeks")

# DB setup
conn = sqlite3.connect("greeks_log.db", check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS greek_log (
        timestamp TEXT,
        symbol TEXT,
        strike INTEGER,
        delta REAL,
        iv REAL,
        theta REAL,
        gamma REAL,
        vega REAL
    )
''')
conn.commit()

# Fetch + log + alert
try:
    greeks = fetch_option_greeks(symbol, strike, option_type, tokens)
    st.subheader("🧠 Option Greeks (updated every 10s)")
    st.json(greeks)

    # Insert to DB
    c.execute('''
        INSERT INTO greek_log VALUES (?,?,?,?,?,?,?,?)
    ''', (
        dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        greeks["symbol"],
        greeks["strike"],
        greeks["delta"],
        greeks["iv"],
        greeks["theta"],
        greeks["gamma"],
        greeks["vega"]
    ))
    conn.commit()

    # Telegram alert logic
    delta_thresh = 0.7
    iv_thresh = 30
    if abs(greeks["delta"]) > delta_thresh or greeks["iv"] > iv_thresh:
        alert_msg = f"⚠️ Alert for {greeks['symbol']}:\nDelta: {greeks['delta']}, IV: {greeks['iv']}%"
        tg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": alert_msg}
        requests.post(tg_url, data=payload)

except Exception as e:
    st.error(f"Live data fetch error: {e}")

# Plot last 10 entries
st.subheader("📈 Delta / IV Chart")
df = pd.read_sql_query("SELECT * FROM greek_log WHERE symbol = ? ORDER BY timestamp DESC LIMIT 10", conn, params=(symbol,))
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

if not df.empty:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(df["timestamp"], df["delta"], label="Delta", marker="o")
    ax2.plot(df["timestamp"], df["iv"], label="IV", color='orange', marker="x")
    ax1.set_ylabel("Delta")
    ax2.set_ylabel("IV")
    ax1.set_xlabel("Time")
    st.pyplot(fig)

