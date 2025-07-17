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
    if not row.empty:
        return str(row.iloc[0]["token"])
    return None

# Get multiple tokens

def get_strike_tokens(symbol, option_type, strikes):
    df = load_master_contract()
    tokens = {}
    for strike in strikes:
        row = df[(df["symbol"] == symbol) & (df["strike"] == strike) & (df["optiontype"] == option_type)]
        if not row.empty:
            tokens[strike] = str(row.iloc[0]["token"])
    return tokens

