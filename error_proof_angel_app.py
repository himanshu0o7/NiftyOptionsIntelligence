# error_proof_angel_app.py (Updated: Uses MasterContract API instead of ScripMaster JSON)

try:
    import streamlit as st
except ModuleNotFoundError:
    raise ImportError("The 'streamlit' module is not installed. Please install it using 'pip install streamlit'.")

try:
    import pandas as pd
except ModuleNotFoundError:
    raise ImportError("The 'pandas' module is required. Please install it using 'pip install pandas'.")

import os
from datetime import datetime
from dotenv import load_dotenv
from SmartApi.smartConnect import SmartConnect
from SmartApi.smartExceptions import DataException

# ---------------------
# Setup
# ---------------------
load_dotenv()
API_KEY = os.getenv("ANGEL_API_KEY")
CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
PIN = os.getenv("ANGEL_PIN")
TOTP = os.getenv("ANGEL_TOTP_SECRET")

@st.cache_resource(show_spinner=False)
def angel_login():
    try:
        obj = SmartConnect(api_key=API_KEY)
        session = obj.generateSession(CLIENT_ID, PIN, TOTP)
        return obj
    except Exception as e:
        st.error(f"❌ Login failed: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_master_contract(smart):
    try:
        df = pd.DataFrame(smart.getMasterContract("NFO"))
        df = df[df['symbol'] == 'NIFTY']  # Only NIFTY contracts
        return df
    except Exception as e:
        st.error(f"❌ Failed to load contract data: {e}")
        return pd.DataFrame()

# ---------------------
# Fetch Greeks Safely
# ---------------------
def get_option_greeks(smart_obj, token):
    try:
        res = smart_obj.optionGreek({"exchange": "NFO", "symboltoken": token})
        return res.get("data", [])
    except Exception as e:
        st.error(f"⚠️ Error fetching Greeks: {e}")
        return []

# ---------------------
# Streamlit UI
# ---------------------
def main():
    st.set_page_config(page_title="Angel One Option Greeks", layout="wide")
    st.title("🧠 Angel One Option Chain + Greeks (MasterContract API)")

    smart = angel_login()
    if not smart:
        st.stop()

    df = load_master_contract(smart)
    if df.empty:
        st.stop()

    expiry_list = sorted(df['expiry'].unique())
    expiry = st.selectbox("📅 Select Expiry:", expiry_list)

    df_filtered = df[df['expiry'] == expiry]
    strike_list = sorted(df_filtered['strike'].unique())
    strike = st.selectbox("🎯 Select Strike Price:", strike_list)

    opt_type = st.radio("Option Type:", ["CE", "PE"], horizontal=True)
    match = df_filtered[
        (df_filtered['strike'] == strike) &
        (df_filtered['optiontype'] == opt_type)
    ]

    if not match.empty:
        token = match.iloc[0]['token']
        st.success(f"✅ SymbolToken: {token}")

        if st.button("🔄 Fetch Greeks"):
            greeks = get_option_greeks(smart, token)
            if greeks:
                st.dataframe(pd.DataFrame(greeks))
            else:
                st.warning("No Greek data returned")
    else:
        st.error("❌ No matching token for selection")

if __name__ == "__main__":
    main()

