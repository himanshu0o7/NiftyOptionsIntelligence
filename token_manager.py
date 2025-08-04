# modules/token_finder.py
# Use this for searching option tokens dynamically from scrip master

import json
import logging
import os

import pandas as pd
import requests
# codex/wrap-requests.get-in-try/except
from requests.exceptions import RequestException
 
# fix-bot-2025-07-24

from telegram_alerts import send_telegram_alert

SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)
LOCAL_SCRIP_FILE = "scrip_master.json"

logger = logging.getLogger(__name__)
# codex/wrap-requests.get-in-try/except


def download_scrip_master(retries: int = 3) -> bool:
    """Download the scrip master with retry and alerting."""
    if os.path.exists(LOCAL_SCRIP_FILE):
        logger.info("token_manager: Using existing scrip master.")
 
_scrip_data_cache = None


def download_scrip_master(retries: int = 3) -> bool:
    """Download the scrip master file with retry and error handling."""
    if os.path.exists(LOCAL_SCRIP_FILE):
        logger.info("TokenManager: Using existing scrip master.")
#fix-bot-2025-07-24
        return True

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(SCRIP_MASTER_URL, timeout=10)
            response.raise_for_status()
# codex/wrap-requests.get-in-try/except
            with open(LOCAL_SCRIP_FILE, "w") as f:
                json.dump(response.json(), f)
            logger.info("token_manager: Scrip master downloaded.")
            return True
        except RequestException as e:
            logger.warning(
                "token_manager: download attempt %s failed: %s", attempt, e
            )
        except OSError as e:
            logger.error("token_manager: Failed to write scrip master: %s", e)
            send_telegram_alert(
                f"token_manager: file save failed - {e}"
            )
            return False

    msg = (
        f"token_manager: Failed to download scrip master after {retries} attempts."
    )
    logger.error(msg)
    send_telegram_alert(msg)
    return False

def load_scrip_data():
    if not download_scrip_master() and not os.path.exists(LOCAL_SCRIP_FILE):
        raise FileNotFoundError("token_manager: scrip master unavailable.")
    with open(LOCAL_SCRIP_FILE, "r") as f:
        return pd.DataFrame(json.load(f))
 
            try:
                with open(LOCAL_SCRIP_FILE, "w", encoding="utf-8") as f:
                    json.dump(response.json(), f)
                logger.info("TokenManager: Scrip master downloaded.")
                return True
            except (FileNotFoundError, PermissionError, IOError) as err:
                logger.error("TokenManager: File write failed: %s", err)
                send_telegram_alert(
                    f"TokenManager: File write failed: {err}"
                )
                return False
        except requests.exceptions.RequestException as err:
            logger.warning(
                "TokenManager: Download attempt %s failed: %s", attempt, err
            )

    send_telegram_alert(
        "TokenManager: Failed to download scrip master after retries."
    )
    logger.error(
        "TokenManager: Download failed after %s attempts. Operating offline.",
        retries,
    )
    return False


def load_scrip_data() -> pd.DataFrame:
    """Load scrip data into a DataFrame.

    Returns empty DataFrame if download fails.
    Uses in-memory cache to avoid redundant file/network operations.
    """
    global _scrip_data_cache
    if _scrip_data_cache is not None:
        return _scrip_data_cache
    if download_scrip_master():
        try:
            with open(LOCAL_SCRIP_FILE, "r", encoding="utf-8") as f:
                _scrip_data_cache = pd.DataFrame(json.load(f))
                return _scrip_data_cache
        except (OSError, json.JSONDecodeError) as err:
            logger.error("TokenManager: Failed to read scrip master: %s", err)
            send_telegram_alert(
                f"TokenManager: Failed to read scrip master: {err}"
            )
    else:
        logger.warning(
            "TokenManager: Scrip master unavailable. Returning empty DataFrame."
        )
    return pd.DataFrame()


def clear_cache() -> None:
    """Reset the in-memory scrip data cache."""
    global _scrip_data_cache
    try:
        _scrip_data_cache = None
        logger.info("TokenManager: cache cleared.")
    except Exception as err:
        logger.error("TokenManager: Failed to clear cache: %s", err)
        send_telegram_alert(f"TokenManager: Failed to clear cache: {err}")

# fix-bot-2025-07-24

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

