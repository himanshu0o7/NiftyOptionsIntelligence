# angelone_master_contract_app.py (Stable, Production-ready)

import os
import json
import pandas as pd
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from SmartApi.smartConnect import SmartConnect
from SmartApi.smartExceptions import DataException

# -----------------------------
# 🔐 Load Environment
# -----------------------------
load_dotenv()
API_KEY = os.getenv("ANGEL_API_KEY")
CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
PIN = os.getenv("ANGEL_PIN")
TOTP = os.getenv("ANGEL_TOTP_SECRET")

# -----------------------------
# 📦 Login with SmartConnect
# -----------------------------
@st.cache_resource(show_spinner=False)
def angel_login():
    try:
        smart = SmartConnect(api_key=API_KEY)
        smart.generateSession(CLIENT_ID, PIN, TOTP)
        return smart
    except Exception as e:
        st.error(f"❌ Login failed: {e}")
        return None

# -----------------------------
# 📂 Load Master Contract
# -----------------------------
def load_master_contract(smart, local_fallback='nfo_master_contract.json'):
    try:
        contract = smart.getMasterContract("NFO")
        with open(local_fallback, "w") as f:
            json.dump(contract, f)
        return pd.DataFrame(contract)
    except Exception as e:
        try:
            st.warning("⚠️ Falling back to local master contract file.")
            with open(local_fallback, "r") as f:
                return pd.DataFrame(json.load(f))
        except:
            st.error(f"❌ Master contract load failed: {e}")
            return pd.DataFrame()

# -----------------------------
# 📈 Option Greeks Fetch
# -----------------------------
def get_option_greeks(smart_obj, token):
    try:
        res = smart_obj.optionGreek({"exchange": "NFO", "symboltoken": token})
        return res.get("data", [])
    except Exception as e:
        st.error(f"⚠️ Error fetching Greeks: {e}")
        return []

# -----------------------------
# 🧠 Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Angel Option Chain Live", layout="wide")
    st.title("📊 Angel One Live Options + Greeks Dashboard")

    smart = angel_login()
    if not smart:
        st.stop()

    df = load_master_contract(smart)
    if df.empty:
        st.stop()

    df = df[df["symbol"] == "NIFTY"]
    expiry_list = sorted(df["expiry"].unique())
    expiry = st.selectbox("📅 Select Expiry", expiry_list)

    df_expiry = df[df["expiry"] == expiry]
    strike_list = sorted(df_expiry["strike"].unique())
    strike = st.selectbox("🎯 Select Strike", strike_list)

    opt_type = st.radio("Option Type", ["CE", "PE"], horizontal=True)
    match = df_expiry[(df_expiry["strike"] == strike) & (df_expiry["optiontype"] == opt_type)]

    if not match.empty:
        token = match.iloc[0]["token"]
        st.code(f"SymbolToken: {token}")

        if st.button("🔍 Fetch Greeks"):
            greeks = get_option_greeks(smart, token)
            if greeks:
                st.dataframe(pd.DataFrame(greeks))
            else:
                st.warning("No Greek data returned")
    else:
        st.error("❌ No match found for your selection")

if __name__ == "__main__":
    main()

