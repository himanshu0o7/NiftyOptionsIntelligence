"""
Standalone runner to stream live option data via SmartAPI's WebSocket V2.

This module encapsulates the steps required to:

1. Load credentials from environment variables (via ``dotenv``).
2. Generate a TOTP and authenticate with Angel One SmartAPI.
3. Retrieve the instrument token for a given option contract.
4. Connect to the WebSocket and print tick data to STDOUT.

You can run this script directly from the command line.  The default
parameters stream LTP quotes for the 25 JUL 2025 NIFTY 25000 call option.

Environment variables required:

* ``ANGEL_API_KEY`` – your API key
* ``ANGEL_CLIENT_ID`` – your client/user ID
* ``ANGEL_PIN`` – your trading PIN/password
* ``ANGEL_TOTP_SECRET`` – the base32 secret used for generating OTPs

Ensure that ``scrip_master_utils.get_token_by_symbol`` is correctly
implemented to map a symbol/expiry/option type/strike into an
instrument token.
"""

import os
import time
import pyotp  # type: ignore
from dotenv import load_dotenv  # type: ignore
from SmartApi import SmartConnect  # type: ignore
from SmartApi.smartWebSocketV2 import SmartWebSocketV2  # type: ignore
from scrip_master_utils import get_token_by_symbol  # type: ignore


def connect_websocket(symbol: str = "NIFTY", expiry: str = "25JUL2025", optiontype: str = "CE", strike: int = 25000) -> None:
    """Connect to the SmartAPI WebSocket and print tick data.

    Parameters
    ----------
    symbol: str
        Underlying symbol (e.g. ``"NIFTY"``).
    expiry: str
        Expiry date in Angel One format (e.g. ``"25JUL2025"``).
    optiontype: str
        Option type (``"CE"`` or ``"PE"``).
    strike: int
        Strike price of the option contract.
    """
    load_dotenv()
    api_key = os.getenv("ANGEL_API_KEY")
    client_code = os.getenv("ANGEL_CLIENT_ID")
    pin = os.getenv("ANGEL_PIN")
    totp_secret = os.getenv("ANGEL_TOTP_SECRET")
    if not all([api_key, client_code, pin, totp_secret]):
        raise ValueError("❌ Missing required environment variables (ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PIN, ANGEL_TOTP_SECRET)")
    # Generate TOTP
    try:
        totp = pyotp.TOTP(totp_secret).now()
    except Exception as exc:
        raise ValueError(f"❌ Invalid ANGEL_TOTP_SECRET (must be base32): {exc}")
    # Login session
    smart_api = SmartConnect(api_key=api_key)
    session = smart_api.generateSession(clientCode=client_code, password=pin, totp=totp)
    feed_token = smart_api.getfeedToken()
    auth_token = session['data']['jwtToken']
    # Fetch instrument token
    token = get_token_by_symbol(symbol, expiry=expiry, optiontype=optiontype, strike=strike)
    if not token:
        raise ValueError("❌ Unable to fetch token for desired option contract.")
    print(f"✅ Logged in successfully. Streaming {symbol} {expiry} {strike}{optiontype} (token {token})")
    # Configure WebSocket
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


if __name__ == "__main__":
    # Default run with typical parameters.  Adjust these as needed or
    # override them by calling ``connect_websocket`` from another module.
    connect_websocket()