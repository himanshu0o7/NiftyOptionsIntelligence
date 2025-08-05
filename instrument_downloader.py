import os
import re
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from utils.logger import Logger


class InstrumentDownloader:
    """Download and process Angel One instrument master data"""

    def __init__(self):
        self.logger = Logger()
        self.instrument_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        self.cache_file = "data/cache/angel_instruments.json"
        self.nifty_tokens_file = "data/cache/nifty_tokens.json"
        self.banknifty_tokens_file = "data/cache/banknifty_tokens.json"
        os.makedirs("data/cache", exist_ok=True)

    def download_and_process(self) -> bool:
        """Download instrument master and extract option tokens"""
        try:
            self.logger.info("Downloading Angel One instrument master...")
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/json'
            }
            response = requests.get(self.instrument_url, headers=headers, timeout=60)
            response.raise_for_status()
            instruments = response.json()

            with open(self.cache_file, 'w') as f:
                json.dump(instruments, f)

            self.logger.info(f"Downloaded {len(instruments)} instruments")

            df = pd.DataFrame(instruments)
            df = df[(df['exch_seg'] == 'NFO') & (df['instrumenttype'] == 'OPTIDX')]
            df = df[df['name'].isin(['NIFTY', 'BANKNIFTY'])]
            df['strike'] = pd.to_numeric(df['strike'], errors='coerce') / 100
            df['strike'] = df['strike'].astype(int)
            df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')
            df = df[df['expiry'] >= datetime.now()]

            def is_weekly_expiry(expiry):
                return (
                    expiry.weekday() == 3 and  # Thursday
                    expiry.month == datetime.now().month and
                    expiry.year == datetime.now().year and
                    timedelta(days=0) <= (expiry - datetime.now()) <= timedelta(days=7)
                )

            df['is_weekly'] = df['expiry'].apply(lambda x: is_weekly_expiry(x) if pd.notnull(x) else False)
            weekly_df = df[df['is_weekly']]

            if weekly_df.empty:
                self.logger.warning("No weekly expiry options found.")
                return False

            # Show available columns before filtering
            expected_cols = ['name', 'symbol', 'expiry', 'strike', 'optiontype', 'tradingsymbol']
            available_cols = [col for col in expected_cols if col in weekly_df.columns]
            print("\n✅ Weekly Expiry Option Tokens:")
            print(weekly_df[available_cols])

            # Save JSON tokens
            if 'name' in weekly_df.columns:
                nifty_df = weekly_df[weekly_df['name'] == 'NIFTY']
                banknifty_df = weekly_df[weekly_df['name'] == 'BANKNIFTY']
                nifty_df.to_json(self.nifty_tokens_file, orient='records')
                banknifty_df.to_json(self.banknifty_tokens_file, orient='records')

            self.logger.info("Saved NIFTY and BANKNIFTY tokens")
            return True

        except Exception as e:
            self.logger.error(f"Error in download_and_process: {str(e)}")
            return False


if __name__ == '__main__':
    downloader = InstrumentDownloader()
    downloader.download_and_process()
nifty_df.to_csv("data/cache/nifty_tokens.csv", index=False)
