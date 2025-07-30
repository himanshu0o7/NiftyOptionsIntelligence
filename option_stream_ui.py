import streamlit as st
import pandas as pd
import time
import threading
import logging
from smart_websocket_handler import SmartWebSocketHandler
from angel_utils import load_nfo_scrip_master
from utils.sensibull_greeks_fetcher import fetch_option_data

logger = logging.getLogger(__name__)

sws = SmartWebSocketHandler()

@st.cache_data(ttl=300)
def get_nifty_option_tokens():
    try:
        df = load_nfo_scrip_master()
        df = df[(df['name'] == 'NIFTY') & (df['instrumenttype'] == 'OPTIDX')]
        if df.empty:
            logger.error("No Nifty options data in scrip master.")
            st.error("No Nifty options data available.")
            return None, None
        
        logger.debug(f"Nifty options DF shape: {df.shape}")
        
        df['expiry_dt'] = pd.to_datetime(df['expiry'], format='%d-%b-%Y', errors='coerce')
        valid_expiries = df['expiry_dt'].dropna().unique()
        if len(valid_expiries) == 0:
            logger.error("No valid expiry dates found.")
            st.error("No valid expiry dates in scrip master.")
            return None, None
        nearest_expiry = sorted(valid_expiries)[0].strftime('%d-%b-%Y')
        
        atm_strike = int(round(df['strike'].mean() / 100) * 100)
        atm_strike_paise = atm_strike * 100
        
        ce_df = df[(df['strike'] == atm_strike_paise) & (df['expiry'] == nearest_expiry) & (df['symbol'].str.endswith('CE'))]
        pe_df = df[(df['strike'] == atm_strike_paise) & (df['expiry'] == nearest_expiry) & (df['symbol'].str.endswith('PE'))]
        
        if ce_df.empty or pe_df.empty:
            logger.error(f"No ATM CE/PE for strike {atm_strike}, expiry {nearest_expiry}. CE shape: {ce_df.shape}, PE shape: {pe_df.shape}")
            st.error(f"No ATM options found for strike {atm_strike}, expiry {nearest_expiry}.")
            return None, None
        
        ce_token = str(ce_df['token'].iloc[0])
        pe_token = str(pe_df['token'].iloc[0])
        logger.info(f"Fetched tokens: CE={ce_token}, PE={pe_token}")
        return ce_token, pe_token
    except Exception as e:
        logger.error(f"Error in get_nifty_option_tokens: {e}", exc_info=True)
        st.error(f"Token fetch error: {e}")
        return None, None

@st.cache_resource
def start_socket():
    ce_token, pe_token = get_nifty_option_tokens()
    if ce_token is None or pe_token is None:
        logger.warning("No valid tokens. WebSocket not started.")
        return []
    token_list = [{"exchangeType": 2, "tokens": [ce_token, pe_token]}]
    sws.start_websocket(token_list=token_list, mode=2)
    return [ce_token, pe_token]

def get_option_data():
    tokens = start_socket()
    if not tokens:
        logger.warning("No tokens available for data fetch.")
        return pd.DataFrame()
    
    data = []
    try:
        greeks_df = fetch_option_data()  # From sensibull_greeks_fetcher
        for token in tokens:
            tick = sws.get_latest_data(token)
            if tick:
                # Merge with Greeks if available
                greeks_row = greeks_df[(greeks_df['strike'] == tick.get('strike', 0)) & (greeks_df['type'] == tick.get('type', ''))]
                greeks_data = greeks_row.to_dict('records')[0] if not greeks_row.empty else {}
                data.append({"Token": token, **tick, **greeks_data})
        df = pd.DataFrame(data)
        logger.debug(f"Option data DF shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error in get_option_data: {e}", exc_info=True)
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

def run_dashboard():
    st.title("Nifty Live CE/PE Stream")
    interval = st.slider("Refresh every n seconds", 5, 60, 10)
    placeholder = st.empty()
    def stream_data():
        while True:
            df = get_option_data()
            with placeholder.container():
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.error("No data available. Check market hours or API.")
            time.sleep(interval)
    thread = threading.Thread(target=stream_data, daemon=True)
    thread.start()

if __name__ == "__main__":
    run_dashboard()

