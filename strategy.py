# strategy.py

from angel_utils import find_token, get_ltp

def check_market_signal(client, symbol, strike, expiry, option_type):
    print(f"🔍 Checking signal for {symbol} {strike} {option_type} {expiry}")
    
    token = find_token(symbol, strike, option_type, expiry)
    if not token:
        print("❌ Token not found")
        return None

    ltp = get_ltp(client, token)
    if ltp is None:
        print("❌ LTP unavailable")
        return None

    # Simple strategy
    if option_type == "CE" and ltp > 100:
        return {"signal": "BUY_CE", "token": token, "ltp": ltp, "reason": "LTP > 100"}

    if option_type == "PE" and ltp > 120:
        return {"signal": "BUY_PE", "token": token, "ltp": ltp, "reason": "LTP > 120"}

    return None
