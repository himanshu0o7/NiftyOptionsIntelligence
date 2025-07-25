"""
Utility functions for logging into Angel One and working with the
NFO scrip master.

This module centralises the handling of environment variables for
Angel One credentials. It exposes helper functions to perform a
login, download the NFO scrip master and query the master for a
particular contract token or last traded price.
"""

import os
import pandas as pd
import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect


# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()

# Support both naming conventions for environment variables
CLIENT_ID = os.getenv("ANGEL_CLIENT_ID") or os.getenv("CLIENT_ID")
PASSWORD = os.getenv("ANGEL_PIN") or os.getenv("PASSWORD")  # Trading PIN or password
TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET") or os.getenv("TOTP")
API_KEY = os.getenv("ANGEL_API_KEY") or os.getenv("SMARTAPI_API_KEY")
SCRIP_MASTER_PATH = "scrip_master.csv"


def login() -> SmartConnect:
    """Perform an authenticated login and return a SmartConnect object.

    Raises a ValueError if any required environment variable is
    missing or if the login fails.
    """
    if not all([CLIENT_ID, PASSWORD, TOTP_SECRET, API_KEY]):
        missing = []
        if not CLIENT_ID:
            missing.append("ANGEL_CLIENT_ID or CLIENT_ID")
        if not PASSWORD:
            missing.append("ANGEL_PIN or PASSWORD")
        if not TOTP_SECRET:
            missing.append("ANGEL_TOTP_SECRET or TOTP")
        if not API_KEY:
            missing.append("ANGEL_API_KEY or SMARTAPI_API_KEY")
        raise ValueError(f"Missing Angel One credentials: {', '.join(missing)}")
    
    # Generate one-time password via TOTP
    try:
        totp = pyotp.TOTP(TOTP_SECRET).now()
    except Exception as e:
        raise ValueError(f"Invalid TOTP_SECRET: {e}")
    
    client = SmartConnect(api_key=API_KEY)
    try:
        session = client.generateSession(clientCode=CLIENT_ID, password=PASSWORD, totp=totp)
        if not session.get("status"):
            import logging
            logging.error(f"Login failed with message: {session.get('message')}")
            raise RuntimeError("Login failed. Please try again later.")
    except Exception as e:
        raise RuntimeError(f"Login failed: {e}")
    
    return client


def load_nfo_scrip_master(client: SmartConnect, force_refresh: bool = False) -> pd.DataFrame:
    """Download and cache the NFO scrip master file.

    If force_refresh is True or the cache file does not exist,
    the master is downloaded using client.get_scrip_master('NFO')
    and written to SCRIP_MASTER_PATH. Otherwise the cached file
    is reused. The returned DataFrame includes the entire NFO master.
    """
    if not os.path.exists(SCRIP_MASTER_PATH) or force_refresh:
        print("⏳ Downloading latest NFO scrip master…")
# fix-bot-2025-07-24
        try:
            df = client.get_scrip_master("NFO")
            df.to_csv(SCRIP_MASTER_PATH, index=False)
            print("✅ NFO scrip master downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download scrip master: {e}")
            raise
    else:
        print("✅ Using cached scrip master")
    
    return pd.read_csv(SCRIP_MASTER_PATH)
=======
        df = client.getMasterContract("NFO")
        df.to_csv(SCRIP_MASTER_PATH, index=False)
    else:
        print("✅ Using cached scrip master")
        df = pd.read_csv(SCRIP_MASTER_PATH)
    return df
 main

# Alias for compatibility
load_master_contract = load_nfo_scrip_master


def find_token(symbol: str, strike: float, option_type: str, expiry: str) -> int | None:
    """Find an instrument token matching the specified criteria.

    The scrip master must already have been downloaded vi
    #fix-bot-2025-07-24
    load_nfo_scrip_master. If the contract is found, its
    token ID is returned as an integer; otherwise None is
    returned.
    """
    if not os.path.exists(SCRIP_MASTER_PATH):
        raise FileNotFoundError("scrip master not found; run load_nfo_scrip_master first")
    
    try:
        df = pd.read_csv(SCRIP_MASTER_PATH)
        filtered = df[
            (df["name"] == symbol)
            & (df["strike"] == float(strike))
            & (df["symbol"].str.endswith(option_type))
            & (df["expiry"] == expiry)
        ]
        
        if not filtered.empty:
            return int(filtered.iloc[0]["token"])
        return None
    except Exception as e:
        print(f"❌ Error finding token: {e}")
        return None
=======
    :func:`load_master_contract`.  If the contract is found, its
    token ID is returned as an integer; otherwise ``None`` is
    returned.
    """
    if not os.path.exists(SCRIP_MASTER_PATH):
        raise FileNotFoundError("scrip master not found; run load_master_contract first")
    df = pd.read_csv(SCRIP_MASTER_PATH)
    filtered = df[
        (df["name"] == symbol)
        & (df["strike"] == float(strike))
        & (df["symbol"].str.endswith(option_type))
        & (df["expiry"] == expiry)
    ]
    if not filtered.empty:
        return int(filtered.iloc[0]["token"])
    return None
 main


def get_ltp(client: SmartConnect, token: int) -> float | None:
    """Retrieve the last traded price (LTP) for a given instrument token."""
    try:
        data = client.ltpData("NFO", token)
        return data['data']['ltp']
    except Exception as exc:
        print(f"❌ LTP Fetch Error: {exc}")
#fix-bot-2025-07-24
        return None
        return None
main
