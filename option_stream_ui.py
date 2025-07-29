import streamlit as st
import pandas as pd
import time
import threading

from smart_websocket_handler import SmartWebSocketHandler
from angel_utils import load_master_contract

sws = SmartWebSocketHandler()

@st.cache_data(ttl=300)
def get_nifty_option_tokens():
    df = load_master_contract()
    df = df[(df['name'] == 'NIFTY') & (df['instrumenttype'] == 'OPTIDX')]
    expiry = sorted(df['expiry'].unique())[0]
    atm_strike = int(round(df['strike'].mean() / 100) * 100)
    ce_token = str(df[(df['strike'] == atm_strike * 100) & (df['expiry'] == expiry) & (df['symbol'].str.endswith('CE'))]['token'].iloc[0])
    pe_token = str(df[(df['strike'] == atm_strike * 100) & (df['expiry'] == expiry) & (df['symbol'].str.endswith('PE'))]['token'].iloc[0])
    return ce_token, pe_token

@st.cache_resource
def start_socket():
    ce_token, pe_token = get_nifty_option_tokens()
    token_list = [{"exchangeType": 2, "tokens": [ce_token, pe_token]}]
    sws.start_websocket(token_list=token_list, mode=2)
    return [ce_token, pe_token]

def get_option_data():
    tokens = start_socket()
    data = []
    for token in tokens:
        tick = sws.get_latest_data(token)
        if tick:
            data.append({"Token": token, **tick})
    return pd.DataFrame(data)

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
            time.sleep(interval)

    thread = threading.Thread(target=stream_data, daemon=True)
    thread.start()

if __name__ == "__main__":
    run_dashboard()