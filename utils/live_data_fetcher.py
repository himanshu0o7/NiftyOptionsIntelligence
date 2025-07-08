"""
Live data fetcher for current week options from Angel One
"""
import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
from utils.logger import Logger

class LiveDataFetcher:
    """Fetch live option data from Angel One instrument master"""
    
    def __init__(self):
        self.logger = Logger()
        self.instrument_url = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
        
    def get_current_expiry_dates(self) -> Dict[str, str]:
        """Get current week expiry dates for NIFTY and BANKNIFTY"""
        today = datetime.now()
        
        # NIFTY expires on Thursday, BANKNIFTY on Wednesday
        def get_next_expiry(base_date, target_weekday):
            days_ahead = target_weekday - base_date.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return base_date + timedelta(days_ahead)
        
        nifty_expiry = get_next_expiry(today, 3)  # Thursday
        banknifty_expiry = get_next_expiry(today, 2)  # Wednesday
        
        return {
            'NIFTY': nifty_expiry.strftime('%d%b%y').upper(),
            'BANKNIFTY': banknifty_expiry.strftime('%d%b%y').upper()
        }
    
    def download_live_instruments(self) -> List[Dict]:
        """Download fresh instrument master from Angel One"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(self.instrument_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to download instruments: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error downloading instruments: {e}")
            return []
    
    def extract_current_options(self, spot_prices: Dict[str, float] = None) -> Dict:
        """Extract current week options with proper strikes"""
        if spot_prices is None:
            spot_prices = {'NIFTY': 23500, 'BANKNIFTY': 51000}  # Default approximates
        
        expiry_dates = self.get_current_expiry_dates()
        instruments = self.download_live_instruments()
        
        if not instruments:
            self.logger.error("No instruments downloaded")
            return {}
        
        options_data = {
            'NIFTY_CE': [],
            'NIFTY_PE': [],
            'BANKNIFTY_CE': [],
            'BANKNIFTY_PE': []
        }
        
        for inst in instruments:
            if inst.get('exch_seg') != 'NFO' or inst.get('instrumenttype') != 'OPTIDX':
                continue
                
            name = inst.get('name', '')
            symbol = inst.get('symbol', '')
            
            # Process NIFTY options
            if 'NIFTY' in name and 'BANKNIFTY' not in name:
                expiry = expiry_dates['NIFTY']
                if expiry in name and (name.endswith('CE') or name.endswith('PE')):
                    strike = self._extract_strike(name)
                    if strike and self._is_valid_strike(strike, spot_prices['NIFTY'], 300):
                        option_type = 'CE' if name.endswith('CE') else 'PE'
                        options_data[f'NIFTY_{option_type}'].append({
                            'symbol': symbol,
                            'name': name,
                            'token': str(inst.get('token')),
                            'strike': strike,
                            'expiry': expiry,
                            'lotsize': inst.get('lotsize', 50),
                            'option_type': option_type
                        })
            
            # Process BANKNIFTY options
            elif 'BANKNIFTY' in name:
                expiry = expiry_dates['BANKNIFTY']
                if expiry in name and (name.endswith('CE') or name.endswith('PE')):
                    strike = self._extract_strike(name)
                    if strike and self._is_valid_strike(strike, spot_prices['BANKNIFTY'], 1000):
                        option_type = 'CE' if name.endswith('CE') else 'PE'
                        options_data[f'BANKNIFTY_{option_type}'].append({
                            'symbol': symbol,
                            'name': name,
                            'token': str(inst.get('token')),
                            'strike': strike,
                            'expiry': expiry,
                            'lotsize': inst.get('lotsize', 15),
                            'option_type': option_type
                        })
        
        # Limit to top 10 options per type (around ATM)
        for key in options_data:
            underlying = 'NIFTY' if 'NIFTY' in key else 'BANKNIFTY'
            spot = spot_prices[underlying]
            
            # Sort by distance from spot price
            options_data[key].sort(key=lambda x: abs(x['strike'] - spot))
            options_data[key] = options_data[key][:10]
        
        return options_data
    
    def _extract_strike(self, name: str) -> int:
        """Extract strike price from option name"""
        import re
        match = re.search(r'(\d+)(?:CE|PE)$', name)
        return int(match.group(1)) if match else None
    
    def _is_valid_strike(self, strike: int, spot: float, range_points: int) -> bool:
        """Check if strike is within reasonable range of spot price"""
        return abs(strike - spot) <= range_points
    
    def save_options_cache(self, options_data: Dict) -> bool:
        """Save options data to cache file"""
        try:
            os.makedirs('data/cache', exist_ok=True)
            with open('data/cache/current_options.json', 'w') as f:
                json.dump(options_data, f, indent=2)
            
            self.logger.info(f"Saved {sum(len(v) for v in options_data.values())} options to cache")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving options cache: {e}")
            return False
    
    def update_live_options(self, spot_prices: Dict[str, float] = None) -> bool:
        """Update options cache with live data"""
        try:
            options_data = self.extract_current_options(spot_prices)
            
            if not options_data or not any(options_data.values()):
                self.logger.error("No options data extracted")
                return False
            
            return self.save_options_cache(options_data)
            
        except Exception as e:
            self.logger.error(f"Error updating live options: {e}")
            return False