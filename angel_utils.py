import os
import logging
from datetime import datetime

import pandas as pd
import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect

from telegram_alerts import send_telegram_alert
from master_contract_fetcher import fetch_and_save_nfo_master_contract

logger = logging.getLogger(__name__)

load_dotenv()

API_KEY = os.getenv("API_KEY")
CLIENT_CODE = os.getenv("CLIENT_CODE")
PASSWORD = os.getenv("PASSWORD")
TOTP_KEY = os.getenv("TOTP_KEY")
SCRIP_MASTER_PATH = "nfo_scrip_master.csv"

class SessionManager:
    def __init__(self):
        self.client = SmartConnect(api_key=API_KEY)
        self.session = None

    def login(self):
        try:
            totp = pyotp.TOTP(TOTP_KEY).now()
            token = self.client.generateSession(
                CLIENT_CODE,
                PASSWORD,
                totp
            )
            self.session = token
            return token
        except Exception as e:
            print(f"Login failed: {e}")
            raise

    def get_session(self):
        if self.session is None:
            return self.login()
        return self.session

    def get_client(self):
        if self.session is None:
            self.login()
        return self.client

def load_nfo_scrip_master(force_refresh: bool = False) -> pd.DataFrame:
    """Load the NFO scrip master contract with cache management.

    Parameters
    ----------
    force_refresh : bool, optional
        When ``True`` the master contract is fetched from the network even if
        a cached file exists.
    """
    try:
        path = SCRIP_MASTER_PATH
        if force_refresh or not os.path.exists(path):
            path = fetch_and_save_nfo_master_contract(path)

        df = pd.read_csv(path)
        if "fetch_timestamp" in df.columns:
            last_fetch = pd.to_datetime(df["fetch_timestamp"].iloc[0])
            if (datetime.now() - last_fetch).days > 1:
                logger.warning("Cached scrip master is stale. Refetching...")
                path = fetch_and_save_nfo_master_contract(path)
                df = pd.read_csv(path)
        else:
            logger.warning("No timestamp in cache. Refetching...")
            path = fetch_and_save_nfo_master_contract(path)
            df = pd.read_csv(path)

        if df.empty:
            raise ValueError("Scrip master is empty.")

        return df
    except Exception as exc:  # noqa: BLE001
        logger.exception("angel_utils: failed to load NFO scrip master: %s", exc)
        try:
            send_telegram_alert(f"angel_utils load_nfo_scrip_master error: {exc}")
        except Exception:  # noqa: BLE001
            logger.exception("angel_utils: failed to send Telegram alert")
        raise


load_master_contract = load_nfo_scrip_master

