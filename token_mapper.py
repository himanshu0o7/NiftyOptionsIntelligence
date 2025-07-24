# fix-bot-2025-07-24
"""
Token mapper for option contracts.

This module downloads the master contract CSV for the NFO segment and
filters it down to index options.  It provides helpers to map a
symbol, strike and option type to a token ID.  The download and
parsing is cached in memory to avoid repeated network calls.
"""

import os
import pandas as pd
import pyotp  # type: ignore
from SmartApi import SmartConnect  # type: ignore
from login_manager import AngelLoginManager  # type: ignore


MASTER_CONTRACT_CACHE: pd.DataFrame | None = None


def load_master_contract(exchange: str = "NFO") -> pd.DataFrame:
    """Load and return the Angel One master contract for the given exchange.

    The master contract is downloaded and cached on the first call.  Only
    index options (``OPTIDX``) for NIFTY and BANKNIFTY are retained.
    """
    global MASTER_CONTRACT_CACHE
    if MASTER_CONTRACT_CACHE is not None:
        return MASTER_CONTRACT_CACHE
    # Use the login manager to ensure we have a valid session
    am = AngelLoginManager()
    session = am.ensure_fresh()
    sc = am.smartconnect
    # Fetch the CSV directly from SmartAPI’s hosted master contracts
    url = f"https://smartapis.in/master-contract/{exchange}.csv"
    df = pd.read_csv(url)
    # Filter for index options
    df = df[df["instrumenttype"] == "OPTIDX"]
    df = df[df["symbol"].isin(["NIFTY", "BANKNIFTY"])]
    # Keep only future expiries
    df["expiry"] = pd.to_datetime(df["expiry"])  # Ensure expiry is in datetime format
    df = df[df["expiry"] >= pd.Timestamp.today()]
    df["strike"] = df["strike"].astype(float).astype(int)
    MASTER_CONTRACT_CACHE = df
    return df


def get_token(symbol: str, strike: int, option_type: str) -> str | None:
    """Return the token ID for a single strike.

    Parameters
    ----------
    symbol: str
        Underlying (``"NIFTY"`` or ``"BANKNIFTY"``)
    strike: int
        Strike price of the option
    option_type: str
        ``"CE"`` or ``"PE"``
    """
    df = load_master_contract()
    row = df[
        (df["symbol"] == symbol)
        & (df["strike"] == strike)
        & (df["optiontype"] == option_type)
    ]
=======
# token_mapper.py (Auto Token Mapping via Angel One Master Contract)

import pandas as pd
from smartapi import SmartConnect
from session_manager import get_access_token, api_key, client_id

MASTER_CONTRACT_CACHE = {}

# Load master contract from Angel One (filter OPTIDX only)
def load_master_contract(exchange="NFO"):
    global MASTER_CONTRACT_CACHE
    if MASTER_CONTRACT_CACHE:
        return MASTER_CONTRACT_CACHE

    sc = SmartConnect(api_key=api_key)
    sc.generate_session(client_id=client_id, access_token=get_access_token())

    url = f"https://smartapis.in/master-contract/{exchange}.csv"
    df = pd.read_csv(url)

    df = df[df["instrumenttype"] == "OPTIDX"]
    df = df[df["symbol"].isin(["NIFTY", "BANKNIFTY"])]
    df = df[df["expiry"] >= pd.Timestamp.today().strftime("%Y-%m-%d")]
    df["strike"] = df["strike"].astype(float).astype(int)

    MASTER_CONTRACT_CACHE = df
    return df

# Get single token for a specific strike

def get_token(symbol, strike, option_type):
    df = load_master_contract()
    row = df[(df["symbol"] == symbol) & (df["strike"] == strike) & (df["optiontype"] == option_type)]
  main
    if not row.empty:
        return str(row.iloc[0]["token"])
    return None

# fix-bot-2025-07-24

def get_strike_tokens(symbol: str, option_type: str, strikes: list[int]) -> dict[int, str]:
    """Return a mapping from strike to token for multiple strikes."""
    df = load_master_contract()
    tokens: dict[int, str] = {}
    for strike in strikes:
        row = df[
            (df["symbol"] == symbol)
            & (df["strike"] == strike)
            & (df["optiontype"] == option_type)
        ]
        if not row.empty:
            tokens[strike] = str(row.iloc[0]["token"])
    return tokens
=======
# Get multiple tokens

def get_strike_tokens(symbol, option_type, strikes):
    df = load_master_contract()
    tokens = {}
    for strike in strikes:
        row = df[(df["symbol"] == symbol) & (df["strike"] == strike) & (df["optiontype"] == option_type)]
        if not row.empty:
            tokens[strike] = str(row.iloc[0]["token"])
    return tokens

  main
