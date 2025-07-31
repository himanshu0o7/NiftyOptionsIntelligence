"""
Real-time instrument data downloader for Angel One API
"""
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from utils.logger import Logger
import os

class InstrumentDownloader:
    """Download and process Angel One instrument master data"""
    
    def __init__(self):
        self.logger = Logger()
        self.instrument_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        self.cache_file = "data/cache/angel_instruments.json"
        self.nifty_tokens_file = "data/cache/nifty_tokens.json"
        self.banknifty_tokens_file = "data/cache/banknifty_tokens.json"
        
        # Create cache directory
        os.makedirs("data/cache", exist_ok=True)
    
    def download_and_process(self) -> bool:
        """Download instrument master and extract option tokens"""
        try:
            self.logger.info("Downloading Angel One instrument master...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Connection': 'keep-alive'
            }
            
            response = requests.get(self.instrument_url, headers=headers, timeout=60)
            response.raise_for_status()
            
            instruments = response.json()
            
            # Save raw data
            with open(self.cache_file, 'w') as f:
                json.dump(instruments, f)
            
            self.logger.info(f"Downloaded {len(instruments)} instruments")
            
            # Process and extract option tokens
            self._extract_option_tokens(instruments)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading instruments: {str(e)}")
            return False
    
    def _extract_option_tokens(self, instruments):
        """Extract NIFTY and BANKNIFTY option tokens"""
        try:
            nifty_options = []
            banknifty_options = []
            
            current_date = datetime.now()
            
            for instrument in instruments:
                symbol = instrument.get('symbol', '')
                name = instrument.get('name', '')
                token = instrument.get('token')
                exchange = instrument.get('exch_seg', '')
                instrument_type = instrument.get('instrumenttype', '')
                
                # Filter for NFO options only
                if exchange == 'NFO' and instrument_type == 'OPTIDX':
                    
                    # NIFTY options
                    if 'NIFTY' in name and 'BANKNIFTY' not in name:
                        # Get current and next week expiry options
                        if self._is_current_week_option(name, current_date):
                            nifty_options.append({
                                'symbol': symbol,
                                'name': name,
                                'token': token,
                                'strike': self._extract_strike(name),
                                'option_type': self._extract_option_type(name),
                                'expiry': self._extract_expiry(name)
                            })
                    
                    # BANKNIFTY options
                    elif 'BANKNIFTY' in name:
                        if self._is_current_week_option(name, current_date):
                            banknifty_options.append({
                                'symbol': symbol,
                                'name': name,
                                'token': token,
                                'strike': self._extract_strike(name),
                                'option_type': self._extract_option_type(name),
                                'expiry': self._extract_expiry(name)
                            })
            
            # Save processed tokens
            with open(self.nifty_tokens_file, 'w') as f:
                json.dump(nifty_options, f, indent=2)
            
            with open(self.banknifty_tokens_file, 'w') as f:
                json.dump(banknifty_options, f, indent=2)
            
            self.logger.info(f"Extracted {len(nifty_options)} NIFTY options and {len(banknifty_options)} BANKNIFTY options")
            
        except Exception as e:
            self.logger.error(f"Error extracting option tokens: {str(e)}")
    
    def _is_current_week_option(self, name: str, current_date: datetime) -> bool:
        """Check if option is for current week or next week"""
        try:
            # Extract date from option name (simplified logic)
            # Real implementation would parse the exact expiry date
            today = current_date.strftime('%d')
            current_month = current_date.strftime('%b').upper()
            
            # Check if option name contains current month
            return current_month in name
            
        except:
            return False
    
    def _extract_strike(self, name: str) -> int:
        """Extract strike price from option name"""
        try:
            # Extract numbers from the end of the name before CE/PE
            import re
            match = re.search(r'(\d+)(?:CE|PE)$', name)
            if match:
                return int(match.group(1))
            return 0
        except:
            return 0
    
    def _extract_option_type(self, name: str) -> str:
        """Extract option type (CE/PE) from name"""
        if name.endswith('CE'):
            return 'CE'
        elif name.endswith('PE'):
            return 'PE'
        return 'UNKNOWN'
    
    def _extract_expiry(self, name: str) -> str:
        """Extract expiry date from option name"""
        try:
            # Simplified extraction - real implementation would be more robust
            import re
            match = re.search(r'(\d{2}[A-Z]{3}\d{2})', name)
            if match:
                return match.group(1)
            return 'UNKNOWN'
        except:
            return 'UNKNOWN'
    
    def get_token_for_symbol(self, symbol: str) -> str:
        """Get token for a given symbol"""
        try:
            # Check NIFTY options
            if os.path.exists(self.nifty_tokens_file):
                with open(self.nifty_tokens_file, 'r') as f:
                    nifty_options = json.load(f)
                
                for option in nifty_options:
                    if option['symbol'] == symbol or option['name'] == symbol:
                        return str(option['token'])
            
            # Check BANKNIFTY options
            if os.path.exists(self.banknifty_tokens_file):
                with open(self.banknifty_tokens_file, 'r') as f:
                    banknifty_options = json.load(f)
                
                for option in banknifty_options:
                    if option['symbol'] == symbol or option['name'] == symbol:
                        return str(option['token'])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting token for symbol {symbol}: {str(e)}")
            return None
    
    def get_current_week_options(self, underlying: str = 'NIFTY', count: int = 10):
        """Get current week options for trading"""
        try:
            file_path = self.nifty_tokens_file if underlying == 'NIFTY' else self.banknifty_tokens_file
            
            if not os.path.exists(file_path):
                return []
            
            with open(file_path, 'r') as f:
                options = json.load(f)
            
            # Sort by strike price and return top options
            options_sorted = sorted(options, key=lambda x: x['strike'])
            return options_sorted[:count]
            
        except Exception as e:
            self.logger.error(f"Error getting current week options: {str(e)}")
            return []

import pandas as pd
from datetime import datetime

# Load CSV
df = pd.read_csv("nfo_scrip_master.csv")

# Filter only NIFTY and BANKNIFTY Options
df = df[
    (df['name'].isin(['NIFTY', 'BANKNIFTY'])) &
    (df['instrumenttype'] == 'OPTIDX')
]

# Convert expiry to datetime
df['expiry'] = pd.to_datetime(df['expiry'], errors='coerce')

# Identify weekly expiry:
# Weekly = upcoming Thursday, NOT monthly
def is_weekly_expiry(row):
    today = datetime.now()
    expiry = row['expiry']
    return (
        expiry.weekday() == 3  # Thursday
        and expiry.month == today.month
        and expiry.year == today.year
        and (expiry - today).days <= 7  # within a week
    )

df['is_weekly'] = df.apply(is_weekly_expiry, axis=1)

# Final filtered result
weekly_df = df[df['is_weekly']]

# Show summary
print("✅ Weekly Expiry Option Tokens:")
print(weekly_df[['name', 'symbol', 'expiry', 'strike', 'optiontype', 'tradingsymbol']].head(20))

