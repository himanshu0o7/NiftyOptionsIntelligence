import streamlit as st
import pandas as pd
import threading
import logging

from utils.sensibull_greeks_fetcher import fetch_option_data
from utils.instrument_downloader import InstrumentDownloader
from utils.ui_refresh import streamlit_autorefresh
from telegram_handler import send_alert

# ─────────────────────────────────────────────────────────────
# ✅ Page and Logger Setup
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Nifty Option Live Dashboard", layout="wide")
st.title("📊 Nifty Options Stream Dashboard")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────────────────────
# ✅ Load Tokens with fallback for optiontype
# ─────────────────────────────────────────────────────────────
def get_nifty_option_tokens():
    try:
        downloader = InstrumentDownloader()
        downloader.download_and_process()

        nifty_df = pd.read_csv("data/cache/nifty_tokens.json")
        banknifty_df = pd.read_json("data/cache/banknifty_tokens.json")

        logger.info(f"NIFTY DF columns: {nifty_df.columns.tolist()}")

        if 'optiontype' not in nifty_df.columns:
            # Create fallback column from symbol
            nifty_df['optiontype'] = nifty_df['symbol'].apply(
                lambda x: 'CE' if x.endswith('CE') else ('PE' if x.endswith('PE') else None)
            )

        ce_df = nifty_df[nifty_df['optiontype'] == 'CE']
        pe_df = nifty_df[nifty_df['optiontype'] == 'PE']
        return ce_df, pe_df

    except Exception as e:
        st.error("❌ Failed to fetch Nifty option tokens.")
        logger.error(f"Error in get_nifty_option_tokens: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────
# ✅ Provide token data to main.py or Streamlit panels
# ─────────────────────────────────────────────────────────────
def get_option_data():
    ce_df, pe_df = get_nifty_option_tokens()
    if ce_df is not None and pe_df is not None:
        return pd.concat([ce_df, pe_df])
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# ✅ Background monitoring with telegram alerts
# ─────────────────────────────────────────────────────────────
def monitor_and_alert():
    df_combined = get_option_data()
    if df_combined.empty:
        return

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


# ─────────────────────────────────────────────────────────────
# ✅ Launch alert background thread
# ─────────────────────────────────────────────────────────────
if "alert_thread" not in st.session_state:
    alert_thread = threading.Thread(target=monitor_and_alert, daemon=True)
    alert_thread.start()
    st.session_state.alert_thread = alert_thread
    st.success("✅ Alert monitoring started in background")


# ─────────────────────────────────────────────────────────────
# ✅ Sidebar UI Refresh Interval Control
# ─────────────────────────────────────────────────────────────
refresh_interval = st.sidebar.selectbox("⏱️ Refresh every", [15, 30, 60], index=1)
streamlit_autorefresh(
    seconds=refresh_interval,
    enable_telegram=True,
    enable_debug_panel=True
)

# ─────────────────────────────────────────────────────────────
# ✅ Strike Picker and OI Chart
# ─────────────────────────────────────────────────────────────
ce_df, pe_df = get_nifty_option_tokens()

if ce_df is not None and pe_df is not None:
    st.subheader("📈 Strike-wise Open Interest (OI)")

    if 'strike' in ce_df.columns and 'oi' in ce_df.columns:
        available_strikes = sorted(set(ce_df['strike']).intersection(set(pe_df['strike'])))
        selected_strikes = st.multiselect(
            "🎯 Select Strikes to Compare", available_strikes, default=available_strikes[:5]
        )

        ce_filtered = ce_df[ce_df['strike'].isin(selected_strikes)]
        pe_filtered = pe_df[pe_df['strike'].isin(selected_strikes)]

        chart_df = pd.DataFrame({
            "Strike": selected_strikes,
            "Call OI": ce_filtered.groupby("strike")["oi"].sum(),
            "Put OI": pe_filtered.groupby("strike")["oi"].sum()
        })

        st.bar_chart(chart_df.set_index("Strike"))
    else:
        st.warning("⚠️ 'strike' or 'oi' column not found in option tokens.")
else:
    st.warning("⚠️ Option data not available for chart rendering.")
