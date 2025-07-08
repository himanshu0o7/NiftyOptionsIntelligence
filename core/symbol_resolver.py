"""
Symbol resolver for Angel One NFO options using official SmartAPI format
"""
import requests
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime
from utils.logger import Logger

class SymbolResolver:
    """Resolve option symbols and tokens using Angel One instrument master"""
    
    def __init__(self):
        self.logger = Logger()
        self.instruments_df = None
        self.instrument_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        
    def load_instruments(self, force_refresh: bool = False) -> bool:
        """Load instrument master from Angel One"""
        if self.instruments_df is not None and not force_refresh:
            return True
            
        try:
            self.logger.info("Downloading Angel One instrument master...")
            response = requests.get(self.instrument_url, timeout=30)
            
            if response.status_code == 200:
                instruments = response.json()
                self.instruments_df = pd.DataFrame(instruments)
                
                # Filter NFO options only
                self.nfo_options = self.instruments_df[
                    (self.instruments_df['exch_seg'] == 'NFO') & 
                    (self.instruments_df['instrumenttype'] == 'OPTIDX')
                ].copy()
                
                # Convert strike to numeric
                self.nfo_options['strike'] = pd.to_numeric(self.nfo_options['strike'], errors='coerce')
                
                self.logger.info(f"Loaded {len(self.nfo_options)} NFO options")
                return True
            else:
                self.logger.error(f"Failed to download instruments: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading instruments: {e}")
            return False
    
    def get_option_symbol_and_token(self, underlying: str, expiry_date: str, 
                                   strike: float, option_type: str) -> tuple[Optional[str], Optional[str]]:
        """Get proper Angel One option symbol and token"""
        if self.instruments_df is None:
            if not self.load_instruments():
                return None, None
        
        try:
            # Angel One stores strikes in lakhs (multiply by 100)
            angel_strike = strike * 100
            
            # Direct search by strike and option type for current expiry
            search_filters = [
                (self.nfo_options['name'] == underlying),
                (self.nfo_options['strike'] == angel_strike)
            ]
            
            # Add option type filter based on symbol ending
            if option_type == 'CE':
                search_filters.append(self.nfo_options['symbol'].str.endswith('CE'))
            else:
                search_filters.append(self.nfo_options['symbol'].str.endswith('PE'))
            
            # Combine all filters
            matching_options = self.nfo_options[
                search_filters[0] & search_filters[1] & search_filters[2]
            ]
            
            if not matching_options.empty:
                # Get the nearest expiry (current week)
                today = datetime.now()
                matching_options = matching_options.copy()
                matching_options['expiry_dt'] = pd.to_datetime(matching_options['expiry'], format='%d%b%Y')
                future_expiries = matching_options[matching_options['expiry_dt'] >= today]
                
                if not future_expiries.empty:
                    # Sort by expiry date and take the nearest
                    nearest = future_expiries.loc[future_expiries['expiry_dt'].idxmin()]
                    self.logger.info(f"Found option: {nearest['symbol']} | Token: {nearest['token']} | Expiry: {nearest['expiry']}")
                    return nearest['symbol'], str(nearest['token'])
            
            # Fallback: find closest available strike price
            underlying_options = self.nfo_options[self.nfo_options['name'] == underlying]
            
            # Filter by option type
            if option_type == 'CE':
                underlying_options = underlying_options[underlying_options['symbol'].str.endswith('CE')]
            else:
                underlying_options = underlying_options[underlying_options['symbol'].str.endswith('PE')]
            
            if not underlying_options.empty:
                # Filter for current week expiries only
                today = datetime.now()
                underlying_options = underlying_options.copy()
                underlying_options['expiry_dt'] = pd.to_datetime(underlying_options['expiry'], format='%d%b%Y')
                current_week = underlying_options[
                    (underlying_options['expiry_dt'] >= today) &
                    (underlying_options['expiry_dt'] <= today + pd.Timedelta(days=7))
                ]
                
                if not current_week.empty:
                    # Find closest strike to our target
                    target_strike = angel_strike
                    current_week['strike_diff'] = abs(current_week['strike'] - target_strike)
                    closest = current_week.loc[current_week['strike_diff'].idxmin()]
                    
                    real_strike = int(closest['strike'] / 100)
                    self.logger.info(f"Found closest strike: {closest['symbol']} | Strike: {real_strike}")
                    return closest['symbol'], str(closest['token'])
            
            self.logger.error(f"No option found for {underlying} {strike} {option_type}")
            return None, None
            
        except Exception as e:
            self.logger.error(f"Error resolving symbol: {e}")
            return None, None
    
    def _format_expiry_for_symbol(self, expiry_date: str) -> str:
        """Convert expiry date to Angel One symbol format"""
        try:
            # Input: "10JUL25" or "09JUL25"
            # Output: "10JUL25" (keep as is for current format)
            
            if len(expiry_date) == 7:  # "10JUL25"
                return expiry_date
            elif len(expiry_date) == 9:  # "10JUL2025"
                # Convert to short format
                return expiry_date[:5] + expiry_date[-2:]
            else:
                return expiry_date
                
        except Exception as e:
            self.logger.error(f"Error formatting expiry date: {e}")
            return expiry_date
    
    def get_current_week_options(self, underlying: str, spot_price: float) -> List[Dict]:
        """Get current week options around spot price"""
        if self.instruments_df is None:
            if not self.load_instruments():
                return []
        
        try:
            # Get current week options
            today = datetime.now()
            
            underlying_options = self.nfo_options[
                self.nfo_options['name'] == underlying
            ].copy()
            
            underlying_options['expiry_dt'] = pd.to_datetime(underlying_options['expiry'])
            
            # Filter current week (next 7 days)
            current_week = underlying_options[
                (underlying_options['expiry_dt'] >= today) &
                (underlying_options['expiry_dt'] <= today + pd.Timedelta(days=7))
            ]
            
            # Get ATM and near ATM strikes
            strikes_range = range(int(spot_price * 0.95), int(spot_price * 1.05), 50)
            
            options_list = []
            for _, option in current_week.iterrows():
                if option['strike'] in strikes_range:
                    options_list.append({
                        'symbol': option['symbol'],
                        'token': str(option['token']),
                        'strike': option['strike'],
                        'option_type': 'CE' if option['symbol'].endswith('CE') else 'PE',
                        'expiry': option['expiry'],
                        'lot_size': option['lotsize']
                    })
            
            return options_list
            
        except Exception as e:
            self.logger.error(f"Error getting current week options: {e}")
            return []
    
    def validate_symbol_token(self, symbol: str, token: str) -> bool:
        """Validate if symbol and token match in instrument master"""
        if self.instruments_df is None:
            if not self.load_instruments():
                return False
        
        try:
            matching = self.nfo_options[
                (self.nfo_options['symbol'] == symbol) &
                (self.nfo_options['token'] == int(token))
            ]
            
            return not matching.empty
            
        except Exception as e:
            self.logger.error(f"Error validating symbol/token: {e}")
            return False