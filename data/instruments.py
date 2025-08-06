import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from utils.logger import Logger
from core.database import Database

class InstrumentManager:
    """Manage instrument master data from Angel One"""

    def __init__(self):
        self.logger = Logger()
        self.db = Database()

        # Angel One instrument master URL
        self.instrument_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

        # Local cache files
        self.cache_dir = "data/cache"
        self.instruments_file = os.path.join(self.cache_dir, "instruments.json")
        self.nifty_options_file = os.path.join(self.cache_dir, "nifty_options.json")
        self.banknifty_options_file = os.path.join(self.cache_dir, "banknifty_options.json")

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        self.instruments_df = None
        self.last_update = None

    def download_instruments(self) -> bool:
        """Download latest instrument master from Angel One"""
        try:
            self.logger.info("Downloading instrument master data...")

            response = requests.get(self.instrument_url, timeout=30)
            response.raise_for_status()

            instruments = response.json()

            # Save to cache file
            with open(self.instruments_file, 'w') as f:
                json.dump(instruments, f, indent=2)

            # Convert to DataFrame
            self.instruments_df = pd.DataFrame(instruments)
            self.last_update = datetime.now()

            self.logger.info(f"Downloaded {len(instruments)} instruments")

            # Extract and cache option data
            self._extract_options_data()

            return True

        except Exception as e:
            self.logger.error(f"Error downloading instruments: {str(e)}")
            return False

    def load_instruments(self, force_download: bool = False) -> bool:
        """Load instruments from cache or download if needed"""
        try:
            # Check if we need to download
            should_download = force_download

            if not should_download and os.path.exists(self.instruments_file):
                # Check file age
                file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.instruments_file))
                if file_age > timedelta(days=1):  # Refresh daily
                    should_download = True
            else:
                should_download = True

            if should_download:
                if not self.download_instruments():
                    # Try to load from cache if download fails
                    if os.path.exists(self.instruments_file):
                        self.logger.warning("Download failed, loading from cache")
                        return self._load_from_cache()
                    return False
            else:
                return self._load_from_cache()

            return True

        except Exception as e:
            self.logger.error(f"Error loading instruments: {str(e)}")
            return False

    def _load_from_cache(self) -> bool:
        """Load instruments from cache file"""
        try:
            if not os.path.exists(self.instruments_file):
                return False

            with open(self.instruments_file, 'r') as f:
                instruments = json.load(f)

            self.instruments_df = pd.DataFrame(instruments)
            self.last_update = datetime.fromtimestamp(os.path.getmtime(self.instruments_file))

            self.logger.info(f"Loaded {len(instruments)} instruments from cache")
            return True

        except Exception as e:
            self.logger.error(f"Error loading from cache: {str(e)}")
            return False

    def _extract_options_data(self):
        """Extract NIFTY and BANKNIFTY options data"""
        try:
            if self.instruments_df is None:
                return

            # Filter NIFTY options
            nifty_options = self.instruments_df[
                (self.instruments_df['name'] == 'NIFTY') &
                (self.instruments_df['instrumenttype'] == 'OPTIDX')
            ].copy()

            # Filter BANKNIFTY options
            banknifty_options = self.instruments_df[
                (self.instruments_df['name'] == 'BANKNIFTY') &
                (self.instruments_df['instrumenttype'] == 'OPTIDX')
            ].copy()

            # Save to cache files
            if not nifty_options.empty:
                nifty_options.to_json(self.nifty_options_file, indent=2, orient='records')
                self.logger.info(f"Cached {len(nifty_options)} NIFTY options")

            if not banknifty_options.empty:
                banknifty_options.to_json(self.banknifty_options_file, indent=2, orient='records')
                self.logger.info(f"Cached {len(banknifty_options)} BANKNIFTY options")

        except Exception as e:
            self.logger.error(f"Error extracting options data: {str(e)}")

    def get_nifty_options(self, expiry_date: str = None, option_type: str = None,
                         strike_range: Tuple[float, float] = None) -> pd.DataFrame:
        """Get NIFTY options with optional filters"""
        try:
            if not os.path.exists(self.nifty_options_file):
                if not self.load_instruments():
                    return pd.DataFrame()

            nifty_options = pd.read_json(self.nifty_options_file)

            if nifty_options.empty:
                return pd.DataFrame()

            # Apply filters
            if expiry_date:
                nifty_options = nifty_options[nifty_options['expiry'] == expiry_date]

            if option_type:
                if option_type.upper() == 'CE':
                    nifty_options = nifty_options[nifty_options['symbol'].str.contains('CE')]
                elif option_type.upper() == 'PE':
                    nifty_options = nifty_options[nifty_options['symbol'].str.contains('PE')]

            if strike_range:
                min_strike, max_strike = strike_range
                nifty_options = nifty_options[
                    (nifty_options['strike'] >= min_strike) &
                    (nifty_options['strike'] <= max_strike)
                ]

            return nifty_options.sort_values('strike')

        except Exception as e:
            self.logger.error(f"Error getting NIFTY options: {str(e)}")
            return pd.DataFrame()

    def get_banknifty_options(self, expiry_date: str = None, option_type: str = None,
                            strike_range: Tuple[float, float] = None) -> pd.DataFrame:
        """Get BANKNIFTY options with optional filters"""
        try:
            if not os.path.exists(self.banknifty_options_file):
                if not self.load_instruments():
                    return pd.DataFrame()

            banknifty_options = pd.read_json(self.banknifty_options_file)

            if banknifty_options.empty:
                return pd.DataFrame()

            # Apply filters
            if expiry_date:
                banknifty_options = banknifty_options[banknifty_options['expiry'] == expiry_date]

            if option_type:
                if option_type.upper() == 'CE':
                    banknifty_options = banknifty_options[banknifty_options['symbol'].str.contains('CE')]
                elif option_type.upper() == 'PE':
                    banknifty_options = banknifty_options[banknifty_options['symbol'].str.contains('PE')]

            if strike_range:
                min_strike, max_strike = strike_range
                banknifty_options = banknifty_options[
                    (banknifty_options['strike'] >= min_strike) &
                    (banknifty_options['strike'] <= max_strike)
                ]

            return banknifty_options.sort_values('strike')

        except Exception as e:
            self.logger.error(f"Error getting BANKNIFTY options: {str(e)}")
            return pd.DataFrame()

    def search_symbol(self, symbol_pattern: str, exchange: str = None,
                     instrument_type: str = None) -> pd.DataFrame:
        """Search for instruments by symbol pattern"""
        try:
            if self.instruments_df is None:
                if not self.load_instruments():
                    return pd.DataFrame()

            # Filter by symbol pattern
            mask = self.instruments_df['symbol'].str.contains(symbol_pattern.upper(), na=False)

            if exchange:
                mask &= (self.instruments_df['exch_seg'] == exchange.upper())

            if instrument_type:
                mask &= (self.instruments_df['instrumenttype'] == instrument_type.upper())

            results = self.instruments_df[mask].copy()

            return results

        except Exception as e:
            self.logger.error(f"Error searching symbol: {str(e)}")
            return pd.DataFrame()

    def get_symbol_info(self, symbol: str, exchange: str = 'NFO') -> Optional[Dict]:
        """Get detailed information for a specific symbol"""
        try:
            if self.instruments_df is None:
                if not self.load_instruments():
                    return None

            symbol_info = self.instruments_df[
                (self.instruments_df['symbol'] == symbol.upper()) &
                (self.instruments_df['exch_seg'] == exchange.upper())
            ]

            if symbol_info.empty:
                return None

            return symbol_info.iloc[0].to_dict()

        except Exception as e:
            self.logger.error(f"Error getting symbol info: {str(e)}")
            return None

    def get_option_chain(self, underlying: str, expiry_date: str) -> Dict[str, pd.DataFrame]:
        """Get complete option chain for an underlying and expiry"""
        try:
            if underlying.upper() == 'NIFTY':
                options_df = self.get_nifty_options(expiry_date=expiry_date)
            elif underlying.upper() == 'BANKNIFTY':
                options_df = self.get_banknifty_options(expiry_date=expiry_date)
            else:
                return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}

            if options_df.empty:
                return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}

            # Separate calls and puts
            calls = options_df[options_df['symbol'].str.contains('CE')].copy()
            puts = options_df[options_df['symbol'].str.contains('PE')].copy()

            return {
                'calls': calls.sort_values('strike'),
                'puts': puts.sort_values('strike')
            }

        except Exception as e:
            self.logger.error(f"Error getting option chain: {str(e)}")
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}

    def get_available_expiries(self, underlying: str) -> List[str]:
        """Get available expiry dates for an underlying"""
        try:
            if underlying.upper() == 'NIFTY':
                options_df = self.get_nifty_options()
            elif underlying.upper() == 'BANKNIFTY':
                options_df = self.get_banknifty_options()
            else:
                return []

            if options_df.empty:
                return []

            expiries = sorted(options_df['expiry'].unique().tolist())
            return expiries

        except Exception as e:
            self.logger.error(f"Error getting expiries: {str(e)}")
            return []

    def get_atm_strikes(self, underlying: str, spot_price: float, expiry_date: str) -> Dict[str, float]:
        """Get ATM strike prices for calls and puts"""
        try:
            option_chain = self.get_option_chain(underlying, expiry_date)

            if option_chain['calls'].empty:
                return {'atm_call': 0, 'atm_put': 0}

            # Find closest strike to spot price
            strikes = sorted(option_chain['calls']['strike'].unique())

            closest_strike = min(strikes, key=lambda x: abs(x - spot_price))

            return {
                'atm_call': closest_strike,
                'atm_put': closest_strike
            }

        except Exception as e:
            self.logger.error(f"Error getting ATM strikes: {str(e)}")
            return {'atm_call': 0, 'atm_put': 0}

    def get_strike_range(self, underlying: str, spot_price: float,
                        range_pct: float = 10) -> Tuple[float, float]:
        """Get strike range around spot price"""
        try:
            range_value = spot_price * (range_pct / 100)
            min_strike = spot_price - range_value
            max_strike = spot_price + range_value

            # Round to nearest 50 for NIFTY, 100 for BANKNIFTY
            if underlying.upper() == 'NIFTY':
                step = 50
            elif underlying.upper() == 'BANKNIFTY':
                step = 100
            else:
                step = 50

            min_strike = round(min_strike / step) * step
            max_strike = round(max_strike / step) * step

            return (min_strike, max_strike)

        except Exception as e:
            self.logger.error(f"Error getting strike range: {str(e)}")
            return (0, 0)

    def update_instruments_cache(self) -> bool:
        """Force update instruments cache"""
        try:
            return self.download_instruments()
        except Exception as e:
            self.logger.error(f"Error updating instruments cache: {str(e)}")
            return False

    def get_cache_status(self) -> Dict:
        """Get cache status information"""
        try:
            status = {
                'instruments_cached': os.path.exists(self.instruments_file),
                'nifty_options_cached': os.path.exists(self.nifty_options_file),
                'banknifty_options_cached': os.path.exists(self.banknifty_options_file),
                'last_update': None,
                'cache_age_hours': None
            }

            if status['instruments_cached']:
                cache_time = datetime.fromtimestamp(os.path.getmtime(self.instruments_file))
                status['last_update'] = cache_time
                status['cache_age_hours'] = (datetime.now() - cache_time).total_seconds() / 3600

            return status

        except Exception as e:
            self.logger.error(f"Error getting cache status: {str(e)}")
            return {}

# Create global instance
instrument_manager = InstrumentManager()
