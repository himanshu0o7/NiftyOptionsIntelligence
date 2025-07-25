# fix-bot-2025-07-24
"""
Utility functions for detecting market trends.

The core function exposed by this module, detect_trend,
analyses option Greek data and open-interest changes to classify the
market as Bullish, Bearish or Sideways. A small wrapper is used to
retrieve delta values either via a live call into
greeks_handler.fetch_option_greeks (if available) or, if that
dependency is absent, by falling back to a sensible default. The
open-interest change is fetched from utils.oi_data.get_oi_change.

In a real-time deployment you should replace the dummy delta values
returned by get_option_greek_data with real data from your
broker's API.
"""

import logging

# Constants for fallback delta values
CALL_DELTA_THRESHOLD = 0.65  # Representative delta for ATM calls
PUT_DELTA_THRESHOLD = -0.65  # Representative delta for ATM puts

try:
    # Attempt to import the live Greek fetcher. This will fail in
    # environments where greeks_handler is unavailable (e.g. when
    # running locally without Angel One connectivity).
    from greeks_handler import fetch_option_greeks
except ImportError:
    fetch_option_greeks = None

try:
    from utils.oi_data import get_oi_change
except ImportError:
    def get_oi_change(symbol: str, expiry: str, option_type: str) -> float:
        """Fallback OI change function"""
        logging.warning("OI data module not available, using dummy data")
        return 100.0  # Dummy positive OI change


def get_option_greek_data(symbol: str, expiry: str, option_type: str, strike: int = None, tokens: Optional[dict] = None) -> dict:
    """Fetch option Greek data for a given contract.

    Parameters
    ----------
    symbol: str
        Underlying index or stock symbol (e.g. "NIFTY").
    expiry: str
        Expiry date in Angel One format (e.g. "25JUL2025"). This
        parameter is currently unused by the fallback implementation.
    option_type: str
        "CE" for call options or "PE" for put options.
    strike: int | None
        Strike price of the option. If None and the live Greek
        fetcher is available, the function may choose a reasonable
        default or return dummy data.
    tokens: dict | None
        Dictionary of authentication tokens required by the live API.

    Returns
    -------
    dict
        A mapping containing at least the key "delta".

    Notes
    -----
    This wrapper attempts to call fetch_option_greeks from
    greeks_handler if it exists. If either the function is
    unavailable or an error is raised during the call, a deterministic
    dummy delta is returned: positive for calls and negative for puts.

# utils/trend_detector.py
"""
Utility functions for detecting market trends.

The core function exposed by this module, detect_trend(),
analyses option Greek data and open-interest changes to classify the
market as Bullish, Bearish or Sideways.
"""

import logging
import yfinance as yf
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from utils.oi_data import get_oi_change
except ImportError:
    def get_oi_change(symbol, strike, option_type, expiry):
        """Mock OI change data for testing."""
        return 1000 if option_type == "CE" else -500

try:
    from greeks_handler import fetch_option_greeks
except ImportError:
    fetch_option_greeks = None


def get_option_greek_data(symbol: str, expiry: str, option_type: str, strike: int = None, tokens: dict = None) -> dict:
    """Retrieve option Greek data (primarily delta) for the given parameters.
    
    If fetch_option_greeks is available, it will be used to fetch live
    data. Otherwise, dummy data is returned for testing purposes.
 main
    """
    if fetch_option_greeks is not None:
        try:
            return fetch_option_greeks(symbol, expiry, option_type, strike, tokens)
        except Exception as exc:
#fix-bot-2025-07-24
            logging.error(f"Greek fetch error: {exc}")
    
    # Fallback: return a representative delta
    return {"delta": CALL_DELTA_THRESHOLD if option_type.upper() == "CE" else PUT_DELTA_THRESHOLD}


def detect_trend(symbol: str, expiry: str) -> dict:
    """Determine the trend of the market for the given symbol and expiry.

    A bullish trend is signalled when the call option delta exceeds 0.6
    **and** call open-interest change is positive. A bearish trend is
    signalled when the put option delta is less than -0.6 and the put
    open-interest change is positive. Otherwise the market is
    considered sideways.

    Parameters
    ----------
    symbol: str
        Underlying index (e.g. "NIFTY" or "BANKNIFTY").
    expiry: str
        Expiry date in Angel One format.

    Returns
    -------
    dict
        A dictionary with "trend" ("Bullish", "Bearish" or
        "Sideways"), "reason" (explanatory string) and
        "supporting_data" (raw values used in the decision).
    """
    try:
        # Greek deltas
        ce_greeks = get_option_greek_data(symbol, expiry, option_type="CE")
        pe_greeks = get_option_greek_data(symbol, expiry, option_type="PE")
        
        # Open interest changes
        ce_oi_change = get_oi_change(symbol, expiry, option_type="CE")
        pe_oi_change = get_oi_change(symbol, expiry, option_type="PE")
        
        ce_delta = ce_greeks.get("delta")
        pe_delta = pe_greeks.get("delta")
        
        # Determine trend based on delta and OI change

            logger.warning(f"Failed to fetch live Greeks: {exc}")
    
    # Return dummy data for testing/development
    return {
        "delta": 0.7 if option_type == "CE" else -0.7,
        "gamma": 0.01,
        "theta": -0.05,
        "vega": 0.02
    }


def detect_trend(symbol: str, expiry: str) -> dict:
    """Detect market trend based on option delta and OI changes.
    
    Args:
        symbol: The underlying symbol (e.g., "NIFTY", "BANKNIFTY")
        expiry: The expiry date string
    
    Returns:
        dict: {
            "trend": "Bullish/Bearish/Sideways/Error",
            "reason": "explanation of the trend detection",
            "supporting_data": { delta and OI data }
        }
    """
    try:
        # Get current price using yfinance
        ticker_map = {
            "NIFTY": "^NSEI",
            "BANKNIFTY": "^NSEBANK"
        }
        
        ticker = ticker_map.get(symbol, "^NSEI")
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        
        if hist.empty:
            raise ValueError(f"No data found for {symbol}")
        
        current_price = hist['Close'].iloc[-1]
        
        # Calculate ATM strike
        if symbol == "NIFTY":
            atm_strike = round(current_price / 50) * 50
        else:  # BANKNIFTY
            atm_strike = round(current_price / 100) * 100
        
        # Fetch CE and PE data
        ce_data = get_option_greek_data(symbol, expiry, "CE", atm_strike)
        pe_data = get_option_greek_data(symbol, expiry, "PE", atm_strike)
        
        # Get OI changes
        ce_oi_change = get_oi_change(symbol, atm_strike, "CE", expiry)
        pe_oi_change = get_oi_change(symbol, atm_strike, "PE", expiry)
        
        ce_delta = ce_data.get("delta")
        pe_delta = pe_data.get("delta")
        
        supporting_data = {
            "ce_delta": ce_delta,
            "pe_delta": pe_delta,
            "ce_oi_change": ce_oi_change,
            "pe_oi_change": pe_oi_change,
            "strike": atm_strike,
            "current_price": current_price
        }
        
        # Trend detection logic
 main
        if (
            ce_delta is not None
            and ce_oi_change is not None
            and ce_delta > 0.6
            and ce_oi_change > 0
        ):
            trend = "Bullish"
            reason = f"CE delta ({ce_delta:.2f}) > 0.6 and CE OI change ({ce_oi_change:,}) > 0"
        elif (
            pe_delta is not None
            and pe_oi_change is not None
            and pe_delta < -0.6
            and pe_oi_change > 0
        ):
            trend = "Bearish"
            reason = f"PE delta ({pe_delta:.2f}) < -0.6 and PE OI change ({pe_oi_change:,}) > 0"
        else:
            trend = "Sideways"
#fix-bot-2025-07-24
            reason = "No strong CE/PE delta + OI change trigger"

        return {
            "trend": trend,
            "reason": reason,
            "supporting_data": {
                "ce_delta": ce_delta,
                "ce_oi_change": ce_oi_change,
                "pe_delta": pe_delta,
                "pe_oi_change": pe_oi_change,
            },
=======
            reason = "Delta and OI conditions not met for clear directional trend"
        
        return {
            "trend": trend,
            "reason": reason,
            "supporting_data": supporting_data
 main
        }
        
    except Exception as exc:
        logger.error(f"Error in trend detection: {exc}")
        return {
            "trend": "Error",
#fix-bot-2025-07-24
            "reason": f"An unexpected error occurred: {exc}",
            "supporting_data": {},

            "reason": f"Failed to analyze trend: {exc}",
            "supporting_data": {}
 main
        }