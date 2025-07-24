import os
import json
import requests
import pandas as pd
from datetime import datetime

SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
LOCAL_SCRIP_FILE = "scrip_master.json"

def download_scrip_master():
    """Download scrip master if not available."""
    if not os.path.exists(LOCAL_SCRIP_FILE):
        print("📥 Downloading scrip master...")
        response = requests.get(SCRIP_MASTER_URL)
        if response.status_code == 200:
            with open(LOCAL_SCRIP_FILE, 'w') as f:
                json.dump(response.json(), f)
            print("✅ Scrip master saved.")
        else:
            raise Exception("❌ Failed to download scrip master.")
    else:
        print("✅ Scrip master already exists.")

def load_scrip_data():
    """Load scrip master into DataFrame and clean expiry."""
    download_scrip_master()
    with open(LOCAL_SCRIP_FILE, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Clean expiry column
    if 'expiry' in df.columns:
        df['expiry'] = df['expiry'].replace('', pd.NA)
        df = df.dropna(subset=['expiry'])
        df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce', format='%d-%b-%Y')
    else:
        raise ValueError("❌ 'expiry' column missing from scrip master.")
    
    return df

def normalize_expiry(expiry_str):
    """Convert '25JUL2025' to datetime.date('2025-07-25')"""
    try:
        return datetime.strptime(expiry_str, '%d%b%Y').date()
    except Exception:
        raise ValueError("❌ Invalid expiry format. Use 'DDMMMYYYY' like '25JUL2025'.")

def get_token_by_symbol(symbol, expiry, strike, optiontype, exchange="NFO", instrumenttype="OPTIDX"):
    df = load_scrip_data()

    normalized_expiry = normalize_expiry(expiry)

    # Convert expiry in df to date
    df['expiry_date'] = df['expiry'].dt.date

    # Apply filters
    df_filtered = df[
        (df['symbol'].str.upper() == symbol.upper()) &
        (df['exch_seg'].str.upper() == exchange.upper()) &
        (df['instrumenttype'].str.upper() == instrumenttype.upper()) &
        (df['expiry_date'] == normalized_expiry) &
        (df['strike'] == float(strike)) &
        (df['symbol'].str.upper().str.endswith(optiontype.upper()))
    ]

    if not df_filtered.empty:
        token = str(df_filtered.iloc[0]['token'])
        print(f"✅ Token found: {token}")
        return token
    else:
        print(f"❌ No token found for: {symbol} {strike} {optiontype} {expiry}")
        available = df[
            (df['symbol'].str.upper() == symbol.upper()) &
            (df['instrumenttype'].str.upper() == instrumenttype.upper())
        ]['expiry_date'].dropna().unique()
        print("🧪 Available Expiries:", sorted(available))
        return None

# Optional debug test
if __name__ == "__main__":
    token = get_token_by_symbol("NIFTY", expiry="25JUL2025", strike=25000, optiontype="CE")
    print("🎯 Final Token:", token)

