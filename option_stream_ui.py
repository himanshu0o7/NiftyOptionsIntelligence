import streamlit as st
import pandas as pd
import time
import numpy as np
from datetime import datetime
import scipy.stats as stats
from scipy.optimize import brentq
import requests
from smart_websocket_handler import SmartWebSocketHandler
from angel_utils import load_master_contract
from streamlit_autorefresh import st_autorefresh

sws = SmartWebSocketHandler()

# Black-Scholes Greeks and IV

def black_scholes_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

def implied_vol(price, S, K, T, r, is_call=True):
    def f(sigma):
        return (black_scholes_call if is_call else black_scholes_put)(S, K, T, r, sigma) - price
    return brentq(f, 1e-6, 5.0)

def send_telegram_message(bot_token, chat_id, message):
    if bot_token and chat_id:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text={message}"
        requests.get(url)

@st.cache_data(ttl=300)
def load_master_df():
    return load_master_contract()

@st.cache_data(ttl=300)
def get_nifty_tokens():
    df = load_master_df()
    df = df[df['name'] == 'NIFTY']
    expiry_list = sorted(df['expiry'].unique())
    expiry = expiry_list[0]
    opt_df = df[(df['expiry'] == expiry) & (df['instrumenttype'] == 'OPTIDX')]
    index_df = df[(df['symbol'] == 'NIFTY') & (df['instrumenttype'] == '')]
    index_token = str(index_df['token'].iloc[0])

    temp_ws = SmartWebSocketHandler()
    temp_ws.start_websocket([{"exchangeType": 1, "tokens": [index_token]}], mode=2)
    time.sleep(2)

    spot = None
    for _ in range(10):
        tick = temp_ws.get_latest_data(index_token)
        if tick and 'ltp' in tick:
            spot = tick['ltp'] / 100.0
            break
        time.sleep(0.5)

    if hasattr(temp_ws, 'close_websocket'):
        temp_ws.close_websocket()

    if not spot:
        raise ValueError("Failed to fetch Nifty spot")

    atm = round(spot / 50) * 50 * 100
    ce_token = str(opt_df[(opt_df['strike'] == atm) & (opt_df['symbol'].str.endswith('CE'))]['token'].iloc[0])
    pe_token = str(opt_df[(opt_df['strike'] == atm) & (opt_df['symbol'].str.endswith('PE'))]['token'].iloc[0])
    return ce_token, pe_token, index_token

def run_dashboard():
    st.title("Nifty Live CE/PE Dashboard")
    bot_token = st.text_input("Telegram Bot Token", type="password")
    chat_id = st.text_input("Telegram Chat ID")

    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'previous_data' not in st.session_state:
        st.session_state.previous_data = None

    col1, col2 = st.columns(2)
    if col1.button("Start"):
        st.session_state.running = True
    if col2.button("Stop"):
        st.session_state.running = False
        if hasattr(sws, 'close_websocket'):
            sws.close_websocket()

    if st.session_state.running:
        try:
            ce_token, pe_token, index_token = get_nifty_tokens()
            sws.start_websocket([
                {"exchangeType": 1, "tokens": [index_token]},
                {"exchangeType": 2, "tokens": [ce_token, pe_token]}
            ], mode=2)
        except Exception as e:
            st.error(f"Start failed: {e}")
            return

        interval = st.slider("Refresh (sec)", 5, 60, 10)
        st_autorefresh(interval=interval * 1000, key="refresh")

        try:
            data = []
            for token in [ce_token, pe_token]:
                tick = sws.get_latest_data(token)
                if tick:
                    data.append({"Token": token, "LTP": tick.get("ltp"), "OI": tick.get("openInterest")})
            df = pd.DataFrame(data)

            if df.empty:
                st.warning("No option data yet")
                return

            master = load_master_df()
            spot_tick = sws.get_latest_data(index_token)
            S = spot_tick['ltp'] / 100.0 if spot_tick else None
            r = 0.07
            now = datetime.now()

            for i, row in df.iterrows():
                try:
                    info = master[master['token'] == int(row['Token'])].iloc[0]
                    K = info['strike'] / 100.0
                    T = max((datetime.strptime(info['expiry'], '%d-%b-%Y') - now).days / 365, 1e-5)
                    price = row['LTP'] / 100.0 if row['LTP'] else 0
                    is_call = info['symbol'].endswith('CE')

                    iv = implied_vol(price, S, K, T, r, is_call)
                    d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
                    d2 = d1 - iv * np.sqrt(T)
                    delta = stats.norm.cdf(d1) if is_call else stats.norm.cdf(d1) - 1
                    gamma = stats.norm.pdf(d1) / (S * iv * np.sqrt(T))
                    vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100
                    theta = -(S * stats.norm.pdf(d1) * iv / (2 * np.sqrt(T)))
                    if is_call:
                        theta -= r * K * np.exp(-r * T) * stats.norm.cdf(d2)
                    else:
                        theta += r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
                    theta /= 365

                    df.at[i, 'IV'] = iv * 100
                    df.at[i, 'Delta'] = delta
                    df.at[i, 'Gamma'] = gamma
                    df.at[i, 'Vega'] = vega
                    df.at[i, 'Theta'] = theta
                except:
                    df.at[i, 'IV'] = np.nan
                    df.at[i, 'Delta'] = np.nan
                    df.at[i, 'Gamma'] = np.nan
                    df.at[i, 'Vega'] = np.nan
                    df.at[i, 'Theta'] = np.nan

            st.dataframe(df[['Token', 'LTP', 'OI', 'IV', 'Delta', 'Gamma', 'Vega', 'Theta']], use_container_width=True)

            if st.session_state.previous_data is not None:
                prev = st.session_state.previous_data
                for i, row in df.iterrows():
                    match = prev[prev['Token'] == row['Token']]
                    if not match.empty:
                        diff = abs(row['OI'] - match['OI'].iloc[0])
                        if diff > 1000:
                            msg = f"Alert: OI change for {row['Token']} → {diff}"
                            send_telegram_message(bot_token, chat_id, msg)

            st.session_state.previous_data = df[['Token', 'OI']].copy()
        except Exception as e:
            st.error(f"Data error: {e}")

if __name__ == "__main__":
    run_dashboard()

