# modules/token_finder.py
# Use this for searching option tokens dynamically from scrip master

import requests
import json
import os
import pandas as pd

SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
LOCAL_SCRIP_FILE = "scrip_master.json"

def download_scrip_master():
    if not os.path.exists(LOCAL_SCRIP_FILE):
        response = requests.get(SCRIP_MASTER_URL)
        if response.status_code == 200:
            with open(LOCAL_SCRIP_FILE, 'w') as f:
                json.dump(response.json(), f)
            print("✅ Scrip master downloaded.")
        else:
            raise Exception(f"❌ Failed to download: {response.status_code}")
    else:
        print("📁 Using existing scrip master.")

def load_scrip_data():
    download_scrip_master()
    with open(LOCAL_SCRIP_FILE, 'r') as f:
        return pd.DataFrame(json.load(f))

def get_token_by_symbol(symbol, exchange='NFO', instrumenttype='OPTIDX', expiry=None, optiontype=None, strike=None):
    df = load_scrip_data()
    query = (
        df['symbol'].str.upper() == symbol.upper()
    ) & (
        df['exchange'] == exchange
    ) & (
        df['instrumenttype'] == instrumenttype
    )

    if expiry:
        query &= df['expiry'] == expiry.upper()
    if optiontype:
        query &= df['name'].str.endswith(optiontype.upper())
    if strike is not None:
        query &= df['strike'] == strike * 100

    results = df[query]
    if not results.empty:
        return results.iloc[0]['token']
    else:
        print(f"❌ No token found for {symbol} {expiry} {optiontype} {strike}")
        return None

