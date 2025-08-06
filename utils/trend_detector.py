import logging
import urllib.request

logger = logging.getLogger(__name__)

try:
    from utils.oi_data import get_oi_change
except ImportError:
    def get_oi_change(symbol, strike, option_type, expiry):
        return 0

def fetch_news_sentiment(symbol: str) -> str:
    try:
        url = f"https://news.google.com/rss/search?q={symbol}+stock+market"
        with urllib.request.urlopen(url) as response:
            text = response.read().decode("utf-8").lower()
            if "bearish" in text:
                return "Bearish"
            elif "bullish" in text:
                return "Bullish"
            else:
                return "Neutral"
    except Exception as e:
        logger.warning(f"Could not fetch news sentiment: {e}")
        return "Neutral"

def detect_trend(symbol: str, expiry: str, ce_strike: float, pe_strike: float) -> str:
    try:
        ce_oi_change = get_oi_change(symbol, ce_strike, 'CE', expiry)
        pe_oi_change = get_oi_change(symbol, pe_strike, 'PE', expiry)
        sentiment = fetch_news_sentiment(symbol)

        if ce_oi_change < 0 and pe_oi_change > 0:
            return "Bearish" if sentiment != "Bullish" else "Sideways"
        elif ce_oi_change > 0 and pe_oi_change < 0:
            return "Bullish" if sentiment != "Bearish" else "Sideways"
        else:
            return sentiment

    except Exception as e:
        logger.error(f"Error detecting trend: {e}")
        return "Unknown"

def test_detect_trend():
    def mock_get_oi_change(symbol, strike, option_type, expiry):
        if option_type == 'CE':
            return 100
        elif option_type == 'PE':
            return -50
        return 0

    def mock_sentiment(symbol):
        return "Neutral"

    global get_oi_change, fetch_news_sentiment
    original_get_oi_change = get_oi_change
    original_fetch_news_sentiment = fetch_news_sentiment

    get_oi_change = mock_get_oi_change
    fetch_news_sentiment = mock_sentiment

    assert detect_trend("NIFTY", "2025-08-01", 24000, 24000) == "Bullish"

    def mock_get_oi_change_bearish(symbol, strike, option_type, expiry):
        if option_type == 'CE':
            return -100
        elif option_type == 'PE':
            return 200
        return 0

    get_oi_change = mock_get_oi_change_bearish
    assert detect_trend("NIFTY", "2025-08-01", 24000, 24000) == "Bearish"

    def mock_get_oi_change_sideways(symbol, strike, option_type, expiry):
        return 50

    get_oi_change = mock_get_oi_change_sideways
    assert detect_trend("NIFTY", "2025-08-01", 24000, 24000) == "Neutral"

    get_oi_change = original_get_oi_change
    fetch_news_sentiment = original_fetch_news_sentiment

if __name__ == "__main__":
    test_detect_trend()
