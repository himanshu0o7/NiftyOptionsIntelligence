import streamlit as st
import pandas as pd
import threading
import logging
from utils.sensibull_greeks_fetcher import fetch_option_data
from utils.instrument_downloader import InstrumentDownloader
from utils.ui_refresh import streamlit_autorefresh
from telegram_handler import send_alert

st.set_page_config(page_title="Nifty Option Live Dashboard", layout="wide")
st.title("📊 Nifty Options Stream Dashboard")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def get_nifty_option_tokens():
    try:
        downloader = InstrumentDownloader()
        downloader.download_and_process()

        nifty_df = pd.read_json("data/cache/nifty_tokens.json")
        banknifty_df = pd.read_json("data/cache/banknifty_tokens.json")

        ce_df = nifty_df[nifty_df['optiontype'] == 'CE']
        pe_df = nifty_df[nifty_df['optiontype'] == 'PE']
        return ce_df, pe_df
    except Exception as e:
        st.error("❌ Failed to fetch Nifty option tokens.")
        logger.error(f"Error in get_nifty_option_tokens: {e}")
        return None, None

def monitor_and_alert():
    ce_df, pe_df = get_nifty_option_tokens()
    if ce_df is None or pe_df is None:
        return

    df_combined = pd.concat([ce_df, pe_df])
    for _, row in df_combined.iterrows():
        symbol = row.get('symbol') or row.get('tradingsymbol')
        if not symbol:
            continue

        try:
            data = fetch_option_data([row['token']])
            if data.empty:
                continue

            opt = data.iloc[0]
            delta = opt.get('delta', 0)
            theta = opt.get('theta', 0)
            vega = opt.get('vega', 0)
            volume = opt.get('volume', 0)

            if delta > 0.4 and abs(theta) < 10 and vega > 1.5 and volume > 2000:
                msg = (
                    f"📢 *Option Alert: {symbol}*\n"
                    f"Δ = `{delta:.2f}`, Θ = `{theta:.2f}`, Vega = `{vega:.2f}`, Vol = `{volume}`"
                )
                send_alert(msg)
        except Exception as e:
            logger.warning(f"Skipping symbol {symbol}: {e}")

# 🚀 Background Thread Start (once)
if "alert_thread" not in st.session_state:
    alert_thread = threading.Thread(target=monitor_and_alert, daemon=True)
    alert_thread.start()
    st.session_state.alert_thread = alert_thread
    st.success("✅ Alert monitoring started in background")

# ⏱️ UI Auto Refresh
refresh_interval = st.sidebar.selectbox("⏱️ Refresh every", [15, 30, 60], index=1)
streamlit_autorefresh(seconds=refresh_interval, enable_telegram=True, enable_debug_panel=True)

# ℹ️ You can add token display or charts here using ce_df/pe_df if needed
