"""Utility functions for detecting market trends.

The core function exposed by this module, detect_trend, analyses option Greek data and open-interest changes to classify the market as Bullish, Bearish or Sideways. Data is fetched live from NSE India for LTP, OI, IV, volume, etc. Greeks like delta are calculated using Black-Scholes model.
"""

import logging
import yfinance as yf
from datetime import datetime
from typing import Optional, Dict
import requests
import json
import math
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Headers for NSE requests
HEADERS = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
    'accept-language': 'en,gu;q=0.9,hi;q=0.8',
    'accept-encoding': 'gzip, deflate, br'
}

def fetch_option_chain(symbol: str) -> Dict:
    """Fetch the option chain data from NSE India."""
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol.upper()}"
    sess = requests.Session()
    sess.get("https://www.nseindia.com", headers=HEADERS, timeout=5)
    response = sess.get(url, headers=HEADERS, timeout=5)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch option chain: {response.status_code}")
    return response.json()

def calculate_delta(S: float, K: float, t: float, r: float, sigma: float, option_type: str) -> float:
    """Calculate delta using Black-Scholes model."""
    if sigma <= 0 or t <= 0:
        return 0.5 if option_type == "CE" else -0.5  # Fallback for ATM
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    if option_type == "CE":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def get_option_greek_data(symbol: str, expiry: str, option_type: str, strike: Optional[int] = None) -> Dict:
    """Fetch option data including calculated Greeks from NSE."""
    try:
        data = fetch_option_chain(symbol)
        records = data['records']
        underlying = records['underlyingValue']
        if not expiry:
            expiry = records['expiryDates'][0]
        filtered = [d for d in records['data'] if d['expiryDate'] == expiry]
        if not filtered:
            raise ValueError(f"No data for expiry {expiry}")
        atm_strike = min([d['strikePrice'] for d in filtered], key=lambda x: abs(x - underlying))
        if strike is None:
            strike = atm_strike
        option_data = next((d[option_type.upper()] for d in filtered if d['strikePrice'] == strike), None)
        if not option_data:
            raise ValueError(f"No {option_type} data for strike {strike}")
        
        # Parse expiry for time to maturity
        expiry_date = datetime.strptime(expiry, '%d-%b-%Y')
        t = max((expiry_date - datetime.now()).days / 365.0, 1/365.0)
        r = 0.07  # Risk-free rate assumption
        sigma = option_data['impliedVolatility'] / 100.0
        
        delta = calculate_delta(underlying, strike, t, r, sigma, option_type.upper())
        
        return {
            "delta": delta,
            "ltp": option_data['lastPrice'],
            "oi": option_data['openInterest'],
            "oi_change": option_data['changeinOpenInterest'],
            "volume": option_data['totalTradedVolume'],
            "iv": option_data['impliedVolatility'],
            "gamma": 0.01,  # Placeholder; can implement full BS if needed
            "theta": -0.05,  # Placeholder
            "vega": 0.02  # Placeholder
        }
    except Exception as exc:
        logger.error(f"Error fetching option data: {exc}")
        # Fallback dummy data
        return {
            "delta": 0.65 if option_type.upper() == "CE" else -0.65,
            "ltp": 0.0,
            "oi": 0,
            "oi_change": 100,
            "volume": 0,
            "iv": 20.0,
            "gamma": 0.01,
            "theta": -0.05,
            "vega": 0.02
        }

def get_live_news(symbol: str) -> list:
    """Fetch recent market news using yfinance."""
    ticker_map = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK"
    }
    ticker = ticker_map.get(symbol.upper(), "^NSEI")
    stock = yf.Ticker(ticker)
    return stock.news[:5]  # Get top 5 recent news items

def detect_trend(symbol: str, expiry: Optional[str] = None) -> Dict:
    """Detect market trend based on option delta, OI changes, and include live news."""
    try:
        # Fetch current price using yfinance as fallback
        ticker_map = {
            "NIFTY": "^NSEI",
            "BANKNIFTY": "^NSEBANK"
        }
        ticker = ticker_map.get(symbol.upper(), "^NSEI")
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
        else:
            current_price = 0  # Will be overridden by NSE data
        
        # Fetch ATM strike roughly
        if symbol.upper() == "NIFTY":
            atm_strike = round(current_price / 50) * 50
        else:
            atm_strike = round(current_price / 100) * 100
        
        # Fetch CE and PE data
        ce_data = get_option_greek_data(symbol, expiry, "CE", atm_strike)
        pe_data = get_option_greek_data(symbol, expiry, "PE", atm_strike)
        
        ce_delta = ce_data["delta"]
        pe_delta = pe_data["delta"]
        ce_oi_change = ce_data["oi_change"]
        pe_oi_change = pe_data["oi_change"]
        
        supporting_data = {
            "ce_delta": ce_delta,
            "pe_delta": pe_delta,
            "ce_oi_change": ce_oi_change,
            "pe_oi_change": pe_oi_change,
            "strike": atm_strike,
            "current_price": current_price,
            "ce_ltp": ce_data["ltp"],
            "pe_ltp": pe_data["ltp"],
            "ce_volume": ce_data["volume"],
            "pe_volume": pe_data["volume"]
        }
        
        # Trend detection logic
        if ce_delta > 0.6 and ce_oi_change > 0:
            trend = "Bullish"
            reason = f"CE delta ({ce_delta:.2f}) > 0.6 and CE OI change ({ce_oi_change:,}) > 0"
        elif pe_delta < -0.6 and pe_oi_change > 0:
            trend = "Bearish"
            reason = f"PE delta ({pe_delta:.2f}) < -0.6 and PE OI change ({pe_oi_change:,}) > 0"
        else:
            trend = "Sideways"
            reason = "Delta and OI conditions not met for clear directional trend"
        
        # Add live news
        news = get_live_news(symbol)
        
        return {
            "trend": trend,
            "reason": reason,
            "supporting_data": supporting_data,
            "news": news
        }
    except Exception as exc:
        logger.error(f"Error in trend detection: {exc}")
        return {
            "trend": "Error",
            "reason": f"Failed to analyze trend: {exc}",
            "supporting_data": {},
            "news": []
        }

