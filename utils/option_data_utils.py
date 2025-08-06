# utils/option_data_utils.py

import requests
import time
import logging

# Dummy implementation. Replace with actual logic based on your data source (e.g., Angel One, NSE, etc.)
def fetch_option_data(symbol: str, expiry: str = None):
    """
    Fetch option chain data for a given symbol and expiry.

    Args:
        symbol (str): e.g. 'NIFTY', 'BANKNIFTY'
        expiry (str): e.g. '25JUL2025'. If None, fetch nearest expiry.

    Returns:
        dict: {'CE': [...], 'PE': [...], 'strikePrices': [...], 'timestamp': ...}
    """
    try:
        logging.info(f"Fetching option data for {symbol}, expiry={expiry}")

        # Simulate a delay (in production, this would be an API call)
        time.sleep(1)

        # Dummy data
        dummy_data = {
            'CE': [
                {'strikePrice': 22500, 'lastPrice': 120, 'openInterest': 11000, 'volume': 1000},
                {'strikePrice': 22600, 'lastPrice': 95, 'openInterest': 9500, 'volume': 900},
            ],
            'PE': [
                {'strikePrice': 22500, 'lastPrice': 100, 'openInterest': 9800, 'volume': 870},
                {'strikePrice': 22400, 'lastPrice': 125, 'openInterest': 10300, 'volume': 950},
            ],
            'strikePrices': [22400, 22500, 22600],
            'timestamp': time.time()
        }

        return dummy_data

    except Exception as e:
        logging.error(f"Error fetching option data: {e}")
        return {}

