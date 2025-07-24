# utils/oi_data.py

def fetch_oi_buildup(symbol):
    """
    Mock OI Buildup data for testing.
    Replace with live data logic from Angel One or other sources.
    """
    return {
        "symbol": symbol,
        "long_buildup": 320000,
        "short_buildup": 150000,
        "long_unwinding": 80000,
        "short_covering": 60000
    }
def get_oi_change(symbol, expiry, option_type):
    return 100000 if option_type == "CE" else 20000

