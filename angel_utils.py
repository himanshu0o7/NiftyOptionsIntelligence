#fix-bot-2025-07-24
"""
Utility functions for logging into Angel One and working with the
NFO scrip master.

This module centralises the handling of environment variables for
Angel One credentials.  It exposes helper functions to perform a
login, download the NFO scrip master and query the master for a
particular contract token or last traded price.
"""

import os
import pandas as pd
import pyotp  # type: ignore
from dotenv import load_dotenv  # type: ignore
from SmartApi import SmartConnect  # type: ignore


# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()
CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
PASSWORD = os.getenv("ANGEL_PIN")  # Trading PIN or password
TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET")
API_KEY = os.getenv("ANGEL_API_KEY")
SCRIP_MASTER_PATH = "scrip_master.csv"


def login() -> SmartConnect:
    """Perform an authenticated login and return a SmartConnect object.

    Raises a ``ValueError`` if any required environment variable is
    missing or if the login fails.
    """
    if not all([CLIENT_ID, PASSWORD, TOTP_SECRET, API_KEY]):
        raise ValueError("Missing Angel One credentials in environment variables.")
    # Generate one‑time password via TOTP
    totp = pyotp.TOTP(TOTP_SECRET).now()
    client = SmartConnect(api_key=API_KEY)
    session = client.generateSession(clientCode=CLIENT_ID, password=PASSWORD, totp=totp)
    if not session.get("status"):
        raise RuntimeError(f"Login failed: {session.get('message')}")
    return client


def load_nfo_scrip_master(client: SmartConnect, force_refresh: bool = False) -> pd.DataFrame:
    """Download and cache the NFO scrip master file.

    If ``force_refresh`` is ``True`` or the cache file does not exist,
    the master is downloaded using ``client.get_scrip_master('NFO')``
    and written to ``SCRIP_MASTER_PATH``.  Otherwise the cached file
    is reused.  The returned DataFrame includes the entire NFO master.
    """
    if not os.path.exists(SCRIP_MASTER_PATH) or force_refresh:
        print("⏳ Downloading latest NFO scrip master…")
=======
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
        main
        df = client.get_scrip_master("NFO")
        df.to_csv(SCRIP_MASTER_PATH, index=False)
    else:
        print("✅ Using cached scrip master")
# fix-bot-2025-07-24
        df = pd.read_csv(SCRIP_MASTER_PATH)
    return df


def find_token(symbol: str, strike: float, option_type: str, expiry: str) -> int | None:
    """Find an instrument token matching the specified criteria.

    The scrip master must already have been downloaded via
    :func:`load_nfo_scrip_master`.  If the contract is found, its
    token ID is returned as an integer; otherwise ``None`` is
    returned.
    """
    if not os.path.exists(SCRIP_MASTER_PATH):
        raise FileNotFoundError("scrip master not found; run load_nfo_scrip_master first")
    df = pd.read_csv(SCRIP_MASTER_PATH)
    filtered = df[
        (df["name"] == symbol)
        & (df["strike"] == float(strike))
        & (df["symbol"].str.endswith(option_type))
        & (df["expiry"] == expiry)
=======
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
    main
    ]
    if not filtered.empty:
        return int(filtered.iloc[0]["token"])
    return None

#fix-bot-2025-07-24

def get_ltp(client: SmartConnect, token: int) -> float | None:
    """Retrieve the last traded price (LTP) for a given instrument token."""
    try:
        data = client.ltpData("NFO", token)
        return data['data']['ltp']  # type: ignore[index]
    except Exception as exc:
        print(f"❌ LTP Fetch Error: {exc}")
        return None
=======
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

   main
