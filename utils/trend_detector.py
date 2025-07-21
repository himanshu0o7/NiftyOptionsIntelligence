# utils/trend_detector.py

#from trend_detector import detect_trend

from utils.greeks_handler import get_option_greek_data
from utils.oi_data import get_oi_change

# utils/trend_detector.py

def detect_trend(symbol, expiry):
    """
    Detects market trend (Bullish, Bearish, Sideways) based on CE/PE delta and OI change.
    Returns:
        dict: {
            "trend": "Bullish/Bearish/Sideways/Error",
            "reason": "...",
            "supporting_data": { ... }
        }
    """
    try:
        from utils.greeks_handler import get_option_greek_data
        from utils.oi_data import get_oi_change

        # Fetch CE and PE Greeks (delta)
        ce_greeks = get_option_greek_data(symbol, expiry, option_type="CE")
        pe_greeks = get_option_greek_data(symbol, expiry, option_type="PE")

        # Fetch OI change
        ce_oi_change = get_oi_change(symbol, expiry, option_type="CE")
        pe_oi_change = get_oi_change(symbol, expiry, option_type="PE")

        # Extract data
        ce_delta = ce_greeks.get("delta")
        pe_delta = pe_greeks.get("delta")

        # Logic
        if ce_delta is not None and ce_oi_change is not None and ce_delta > 0.6 and ce_oi_change > 0:
            trend = "Bullish"
            reason = "CE delta > 0.6 and CE OI change > 0"
        elif pe_delta is not None and pe_oi_change is not None and pe_delta < -0.6 and pe_oi_change > 0:
            trend = "Bearish"
            reason = "PE delta < -0.6 and PE OI change > 0"
        else:
            trend = "Sideways"
            reason = "No strong CE/PE delta + OI change trigger"

        return {
            "trend": trend,
            "reason": reason,
            "supporting_data": {
                "ce_delta": ce_delta,
                "ce_oi_change": ce_oi_change,
                "pe_delta": pe_delta,
                "pe_oi_change": pe_oi_change
            }
        }

    except Exception as e:
        return {
            "trend": "Error",
            "reason": str(e),
            "supporting_data": {}
        }

