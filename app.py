import os
import time
import sqlite3
import datetime as dt
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh
import streamlit as st
from login_manager import AngelOneLogin
from greeks_handler import fetch_option_greeks
from utils.oi_data import fetch_oi_buildup

load_dotenv()

# Load .env
API_KEY = os.getenv("ANGEL_API_KEY")
CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
MPIN = os.getenv("ANGEL_PIN")
TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not all([API_KEY, CLIENT_ID, MPIN, TOTP_SECRET]):
    st.error("❌ API credentials not loaded.")
    st.stop()

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

# Login
try:
    login = AngelOneLogin(API_KEY, CLIENT_ID, MPIN, TOTP_SECRET)
    tokens = login.login()
    st.success(f"✅ Logged in as {tokens['clientcode']}")
except Exception as e:
    st.error(f"Login Failed: {e}")
    st.stop()

# UI Controls
st.title("📊 Nifty Options Intelligence Dashboard")
symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"])
strike = st.number_input("Strike Price", value=23500)
option_type = st.radio("Option Type", ["CE", "PE"])

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📡 Auto Mode", "🖱️ Manual Mode", "📈 Chart & Logs", "💬 Alerts"])

# ================================
with tab1:
    st_autorefresh(interval=10000, limit=None, key="autorefresh")
    try:
        greeks = fetch_option_greeks(symbol, strike, option_type, tokens)
        oi_data = fetch_oi_buildup(symbol)

        st.subheader("🧠 Option Greeks")
        st.json(greeks)
        st.subheader("📊 OI Buildup")
        st.json(oi_data)

        # Log
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

        # Telegram alert
        if greeks["delta"] > 0.7 or greeks["iv"] > 30:
            msg = f"⚠️ Alert: {greeks['symbol']} {greeks['strike']} {option_type}\nDelta: {greeks['delta']:.2f}\nIV: {greeks['iv']}%"
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
            st.warning("📣 Telegram alert triggered!")

    except Exception as e:
        st.error(f"Auto fetch error: {e}")

# ================================
with tab2:
    if st.button("🛱️ Fetch Market Data"):
        try:
            greeks = fetch_option_greeks(symbol, strike, option_type, tokens)
            oi_data = fetch_oi_buildup(symbol)
            st.subheader("🧠 Option Greeks")
            st.json(greeks)
            st.subheader("📊 OI Buildup")
            st.json(oi_data)
        except Exception as e:
            st.error(f"Manual fetch error: {e}")

# ================================
with tab3:
    st.subheader("📈 Last 10 Greeks (Delta & IV)")

    df = pd.read_sql_query("SELECT * FROM greek_log WHERE symbol = ? ORDER BY timestamp DESC LIMIT 10", conn, params=(symbol,))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    if not df.empty:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(df["timestamp"], df["delta"], label="Delta", marker="o")
        ax2.plot(df["timestamp"], df["iv"], label="IV", marker="x", color='orange')
        ax1.set_ylabel("Delta")
        ax2.set_ylabel("IV")
        st.pyplot(fig)

    st.subheader("⬇️ Export Full Log")
    df_all = pd.read_sql_query("SELECT * FROM greek_log", conn)
    csv = df_all.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, file_name="greeks_log.csv", mime="text/csv")

# ================================
with tab4:
    st.subheader("💬 Recent Alerts")
    df = pd.read_sql_query("SELECT * FROM greek_log WHERE delta > 0.7 OR iv > 30 ORDER BY timestamp DESC LIMIT 3", conn)
    if df.empty:
        st.info("✅ No recent alerts.")
    else:
        for i, row in df.iterrows():
            st.warning(f"[{row['timestamp']}] {row['symbol']} {row['strike']} {option_type}\nDelta: {row['delta']:.2f} | IV: {row['iv']}%")

