import os
import pandas as pd
import pyotp
import requests
from dotenv import load_dotenv
from SmartApi import SmartConnect

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

def load_nfo_scrip_master():
    if not os.path.exists(SCRIP_MASTER_PATH):
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data)
            nfo_df = df[df['exch_seg'] == 'NFO']
            nfo_df.to_csv(SCRIP_MASTER_PATH, index=False)
            print("✅ NFO scrip master downloaded successfully")
        except Exception as e:
            print(f"❌ Failed to download scrip master: {e}")
            raise
    else:
        print("✅ Using cached scrip master")

    return pd.read_csv(SCRIP_MASTER_PATH)

load_master_contract = load_nfo_scrip_master

def load_nfo_scrip_master():
    if not os.path.exists(SCRIP_MASTER_PATH):
        path = fetch_and_save_nfo_master_contract()  # From master_contract_fetcher.py
    else:
        df = pd.read_csv(SCRIP_MASTER_PATH)
        if 'fetch_timestamp' in df.columns:
            last_fetch = pd.to_datetime(df['fetch_timestamp'].iloc[0])
            if (datetime.now() - last_fetch).days > 1:
                logger.warning("Cached scrip master is stale. Refetching...")
                path = fetch_and_save_nfo_master_contract()
        else:
            logger.warning("No timestamp in cache. Refetching...")
            path = fetch_and_save_nfo_master_contract()
        df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Scrip master is empty.")
    return df

