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
    """
    if fetch_option_greeks is not None:
        try:
            return fetch_option_greeks(symbol, expiry, option_type, strike, tokens)
        except Exception as exc:
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
            reason = "Delta and OI conditions not met for clear directional trend"
        
        return {
            "trend": trend,
            "reason": reason,
            "supporting_data": supporting_data
        }
        
    except Exception as exc:
        logger.error(f"Error in trend detection: {exc}")
        return {
            "trend": "Error",
            "reason": f"Failed to analyze trend: {exc}",
            "supporting_data": {}
        }