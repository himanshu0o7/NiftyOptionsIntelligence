# angel_utils.py

import os
import pandas as pd
import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
PASSWORD = os.getenv("PASSWORD")           # Login PIN or Password
TOTP_SECRET = os.getenv("TOTP")
API_KEY = os.getenv("SMARTAPI_API_KEY")    # Must be added to .env as well
SCRIP_MASTER_PATH = "scrip_master.csv"

# -----------------------------
# Login and return client
# -----------------------------
def login():
    totp = pyotp.TOTP(TOTP_SECRET).now()
    client = SmartConnect(api_key=API_KEY)
    session = client.generateSession(CLIENT_ID, PASSWORD, totp)
    return client

# -----------------------------
# Scrip Master Fetcher
# -----------------------------
def load_nfo_scrip_master(client, force_refresh=False):
    if not os.path.exists(SCRIP_MASTER_PATH) or force_refresh:
        print("⏳ Downloading latest NFO scrip master...")
        df = client.get_scrip_master("NFO")
        df.to_csv(SCRIP_MASTER_PATH, index=False)
    else:
        print("✅ Using cached scrip master")
    return pd.read_csv(SCRIP_MASTER_PATH)

# -----------------------------
# Token Finder
# -----------------------------
def find_token(symbol, strike, option_type, expiry):
    df = pd.read_csv(SCRIP_MASTER_PATH)
    filtered = df[
        (df["name"] == symbol) &
        (df["strike"] == float(strike)) &
        (df["symbol"].str.endswith(option_type)) &
        (df["expiry"] == expiry)
    ]
    if not filtered.empty:
        return int(filtered.iloc[0]["token"])
    return None

# -----------------------------
# LTP Fetcher
# -----------------------------
def get_ltp(client, token):
    try:
        data = client.ltpData("NFO", token)
        return data['data']['ltp']
    except Exception as e:
        print(f"❌ LTP Fetch Error: {e}")
        return None

