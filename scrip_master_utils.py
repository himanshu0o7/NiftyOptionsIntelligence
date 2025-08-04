import json
import logging
import os
from datetime import datetime

import pandas as pd
import requests

from telegram_alerts import send_telegram_alert

SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)
LOCAL_SCRIP_FILE = "scrip_master.json"

logger = logging.getLogger(__name__)


def download_scrip_master(retries: int = 3) -> bool:
    """Download scrip master if not available."""
    if os.path.exists(LOCAL_SCRIP_FILE):
        logger.info("ScripMasterUtils: Scrip master already exists.")
        return True

    logger.info("ScripMasterUtils: Downloading scrip master...")
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(SCRIP_MASTER_URL, timeout=10)
            response.raise_for_status()
            try:
                with open(LOCAL_SCRIP_FILE, "w", encoding="utf-8") as f:
                    json.dump(response.json(), f)
                logger.info("ScripMasterUtils: Scrip master saved.")
                return True
            except (FileNotFoundError, PermissionError, IOError) as err:
                logger.error("ScripMasterUtils: File write failed: %s", err)
                send_telegram_alert(
                    f"ScripMasterUtils: File write failed: {err}"
                )
                return False
        except requests.exceptions.RequestException as err:
            logger.warning(
                "ScripMasterUtils: Download attempt %s failed: %s", attempt, err
            )

    send_telegram_alert(
        "ScripMasterUtils: Failed to download scrip master after retries."
    )
    logger.error("ScripMasterUtils: Failed to download scrip master.")
    return False


def load_scrip_data() -> pd.DataFrame:
    """Load scrip master into DataFrame and clean expiry."""
    if not download_scrip_master():
        logger.warning(
            "ScripMasterUtils: Using empty DataFrame - scrip master unavailable."
        )
        return pd.DataFrame()

    try:
        with open(LOCAL_SCRIP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as err:
        logger.error("ScripMasterUtils: Failed to read scrip master: %s", err)
        send_telegram_alert(
            f"ScripMasterUtils: Failed to read scrip master: {err}"
        )
        return pd.DataFrame()

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

