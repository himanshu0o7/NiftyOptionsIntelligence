# websocket_runner.py - Live LTP feed, login, auto WebSocket connection

import os
import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from scrip_master_utils import get_token_by_symbol

# === Load environment variables ===
load_dotenv()

API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PIN = os.getenv("PIN")
TOTP_SECRET = os.getenv("TOTP_SECRET")

if not all([API_KEY, CLIENT_CODE, PIN, TOTP_SECRET]):
    raise ValueError("❌ Missing required environment variables.")

# === Generate TOTP ===
try:
    totp = pyotp.TOTP(TOTP_SECRET).now()
except Exception as e:
    raise ValueError(f"❌ Invalid TOTP_SECRET (must be base32): {e}")

# === Login Session ===
smart_api = SmartConnect(api_key=API_KEY)
session = smart_api.generateSession(CLIENT_CODE, PIN, totp)
feed_token = smart_api.getfeedToken()
auth_token = session['data']['jwtToken']
refresh_token = session['data']['refreshToken']

print("✅ Logged in successfully")

# === Get instrument token ===
token = get_token_by_symbol("NIFTY", expiry="25JUL2025", optiontype="CE", strike=25000)
if not token:
    raise ValueError("❌ Unable to fetch token for desired option contract.")

print(f"🎯 Token fetched: {token}")

# === Setup WebSocket ===
sws = SmartWebSocketV2(auth_token, API_KEY, CLIENT_CODE, feed_token)
mode = 1  # Mode 1 = LTP only
correlation_id = "kp5feed"

# === Event callbacks ===
def on_open(wsapp):
    print("🟢 WebSocket Connected.")
    sws.subscribe(correlation_id, mode, [{"exchangeType": 2, "tokens": [token]}])

def on_data(wsapp, message):
    print("📈 Tick Data:", message)
    # Optional: trigger alert/trade here

def on_error(wsapp, error):
    print("❌ WebSocket Error:", error)

def on_close(wsapp):
    print("🔌 WebSocket Closed. Reconnecting...")
    # Optional: Retry logic

# === Bind callbacks ===
sws.on_open = on_open
sws.on_data = on_data
sws.on_error = on_error
sws.on_close = on_close

# === Connect ===
sws.connect()

