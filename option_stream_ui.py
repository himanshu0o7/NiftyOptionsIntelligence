# option_stream_ui.py
# Uses REST API (getQuote) instead of WebSocket for safe integration in Streamlit

import os
import pandas as pd
from SmartApi.smartConnect import SmartConnect
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
CLIENT_ID = os.getenv("CLIENT_ID")
PIN = os.getenv("PIN")
TOTP = os.getenv("TOTP")

# Session generator (can be reused)
def get_session():
    try:
        obj = SmartConnect(api_key=API_KEY)
        session_data = obj.generateSession(CLIENT_ID, PIN, TOTP)
        return obj, session_data
    except Exception as e:
        return None, {"error": f"Login/session failed: {e}"}

# Option chain REST-based fetcher
def get_option_data(symbol: str, strike: int, option_type: str) -> dict:
    try:
        obj, session = get_session()
        if not obj:
            return session  # error info

        # Construct tradingsymbol
        # Example: NIFTY24JUL22500CE
        tradingsymbol = f"{symbol}24JUL{strike}{option_type.upper()}"

        quote = obj.getQuote(
            exchange="NFO",
            tradingsymbol=tradingsymbol
        )

        # Return specific fields (can be expanded)
        ltp = quote.get("data", {}).get("fetched", {}).get("last_price")
        return {
            "symbol": tradingsymbol,
            "ltp": ltp,
            "raw": quote
        }

    except Exception as e:
        return {"error": f"Fetch failed: {e}"}

