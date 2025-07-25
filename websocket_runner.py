"""
Standalone runner to stream live option data via SmartAPI's WebSocket V2.

This module encapsulates the steps required to:

1. Load credentials from environment variables (via dotenv).
2. Generate a TOTP and authenticate with Angel One SmartAPI.
3. Retrieve the instrument token for a given option contract.
4. Connect to the WebSocket and print tick data to STDOUT.

You can run this script directly from the command line. The default
parameters stream LTP quotes for the 25 JUL 2025 NIFTY 25000 call option.

Environment variables required:

* ANGEL_API_KEY – your API key
* ANGEL_CLIENT_ID – your client/user ID
* ANGEL_PIN – your trading PIN/password
* ANGEL_TOTP_SECRET – the base32 secret used for generating OTPs

Ensure that scrip_master_utils.get_token_by_symbol is correctly
implemented to map a symbol/expiry/option type/strike into an
instrument token.
"""

import os
import time
import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from scrip_master_utils import get_token_by_symbol


def connect_websocket(symbol: str = "NIFTY", expiry: str = "25JUL2025", optiontype: str = "CE", strike: int = 25000) -> None:
    """Connect to the SmartAPI WebSocket and print tick data.

    Parameters
    ----------
    symbol: str
        Underlying symbol (e.g. "NIFTY").
    expiry: str
        Expiry date in Angel One format (e.g. "25JUL2025").
    optiontype: str
        Option type ("CE" or "PE").
    strike: int
        Strike price of the option contract.
    """
    load_dotenv()
    
    # Get environment variables with fallback to common names
    api_key = os.getenv("ANGEL_API_KEY") or os.getenv("API_KEY")
    client_code = os.getenv("ANGEL_CLIENT_ID") or os.getenv("CLIENT_CODE")
    pin = os.getenv("ANGEL_PIN") or os.getenv("PIN")
    totp_secret = os.getenv("ANGEL_TOTP_SECRET") or os.getenv("TOTP_SECRET")
    
    missing_vars = []
    if not api_key:
        missing_vars.append("ANGEL_API_KEY or API_KEY")
    if not client_code:
        missing_vars.append("ANGEL_CLIENT_ID or CLIENT_CODE")
    if not pin:
        missing_vars.append("ANGEL_PIN or PIN")
    if not totp_secret:
        missing_vars.append("ANGEL_TOTP_SECRET or TOTP_SECRET")
        
    if missing_vars:
        raise ValueError(f"❌ Missing required environment variable(s): {', '.join(missing_vars)}")
    
    # Generate TOTP
    try:
        totp = pyotp.TOTP(totp_secret).now()
    except Exception as exc:
        raise ValueError(f"❌ Invalid TOTP_SECRET (must be base32): {exc}")
    
    # Login session
    try:
        smart_api = SmartConnect(api_key=api_key)
        session = smart_api.generateSession(clientCode=client_code, password=pin, totp=totp)
        feed_token = smart_api.getfeedToken()
        
        if not session or 'data' not in session or 'jwtToken' not in session['data']:
            raise ValueError("❌ Invalid session structure: Missing 'data' or 'jwtToken' in session response.")
        
        auth_token = session['data']['jwtToken']
    except Exception as e:
        raise ValueError(f"❌ Login failed: {e}")
    
    # Fetch instrument token
    try:
        token = get_token_by_symbol(symbol, expiry=expiry, optiontype=optiontype, strike=strike)
        if not token:
            raise ValueError("❌ Unable to fetch token for desired option contract.")
    except Exception as e:
        raise ValueError(f"❌ Token fetch failed: {e}")
    
    print(f"✅ Logged in successfully. Streaming {symbol} {expiry} {strike}{optiontype} (token {token})")
    
    # Configure WebSocket
    try:
        sws = SmartWebSocketV2(auth_token, api_key, client_code, feed_token)
        mode = 1  # Mode 1 = LTP only
        correlation_id = "nifty_ws"
        
        # Event handlers
        def on_open(wsapp):
            print("🟢 WebSocket Connected.")
            sws.subscribe(correlation_id, mode, [{"exchangeType": 2, "tokens": [token]}])
        
        def on_data(wsapp, message):
            print("📈 Tick Data:", message)
        
        def on_error(wsapp, error):
            print("❌ WebSocket Error:", error)
        
        def on_close(wsapp):
            print("🔌 WebSocket Closed. Attempting to reconnect…")
        
        sws.on_open = on_open
        sws.on_data = on_data
        sws.on_error = on_error
        sws.on_close = on_close
        
        # Connect (blocks until terminated)
        sws.connect()
        
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        raise


if __name__ == "__main__":
#fix-bot-2025-07-24
    # Default run with typical parameters. Adjust these as needed or
    # override them by calling connect_websocket from another module.
    try:
        connect_websocket()
    except Exception as e:
        print(f"❌ Error in websocket_runner: {e}")
        exit(1)

    # Default run with typical parameters.  Adjust these as needed or
    # override them by calling ``connect_websocket`` from another module.
    connect_websocket()
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
 main
