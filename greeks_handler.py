# greeks_handler.py

import time
import threading
import datetime as dt
import pandas as pd
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

# Global cache
option_data = {}
symbol_dict = {}
sws = None

def fetch_option_greeks(symbol, strike, option_type, tokens):
    """
    Live Angel One Greek fetcher using SmartAPI's optionGreek API.
    """

    # Setup SmartConnect object
    obj = SmartConnect(api_key=tokens["api_key"])
    obj.jwt_token = tokens["jwtToken"]
    obj.feed_token = tokens["feedToken"]
    obj.user_id = tokens["clientcode"]

    # Fetch instrument list
    import urllib.request, json
    instrument_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    response = urllib.request.urlopen(instrument_url)
    instrument_list = json.loads(response.read())

    # Get valid contracts for symbol (e.g. BANKNIFTY)
    contracts = [i for i in instrument_list if i["name"] == symbol and i["instrumenttype"] in ["OPTIDX", "OPTSTK"]]
    df = pd.DataFrame(contracts)

    if df.empty:
        return {"error": "Symbol not found in instrument list"}

    # Nearest expiry
    df["strike"] = df["strike"].astype(float) / 100
    df["expiry_dt"] = pd.to_datetime(df["expiry"])
    df = df[(df["strike"] == strike) & (df["symbol"].str.endswith(option_type))]
    if df.empty:
        return {"error": f"No {option_type} contract found for {symbol} {strike}"}

    df = df.sort_values("expiry_dt").reset_index(drop=True)
    instrument = df.iloc[0]
    token = instrument["token"]
    expiry = instrument["expiry"]

    # Setup WebSocket listener
    def on_data(wsapp, message):
        option_data[instrument["symbol"]] = {
            "ltp": message["last_traded_price"],
            "oi": message["open_interest"],
            "volume": message["volume_trade_for_the_day"],
            "bid": message["best_5_buy_data"][0]["price"],
            "ask": message["best_5_sell_data"][0]["price"]
        }

    def on_open(wsapp):
        wsapp.subscribe("greeks_stream", 3, [{"exchangeType": 2, "tokens": [token]}])

    def on_error(wsapp, error):
        print("WebSocket error:", error)

    global sws
    sws = SmartWebSocketV2(tokens["jwtToken"], tokens["api_key"], tokens["clientcode"], tokens["feedToken"])
    sws.on_data = on_data
    sws.on_open = on_open
    sws.on_error = on_error

    t = threading.Thread(target=sws.connect, daemon=True)
    t.start()
    time.sleep(3)  # wait for data

    # Get Greeks
    greek_resp = obj.optionGreek({"name": symbol, "expirydate": expiry})
    greeks = [g for g in greek_resp["data"] if float(g["strikePrice"]) == strike and g["optionType"] == option_type]

    if not greeks:
        return {"error": "No matching Greek data"}

    g = greeks[0]
    live = option_data.get(instrument["symbol"], {})

    return {
        "symbol": instrument["symbol"],
        "strike": strike,
        "option_type": option_type,
        "expiry": expiry,
        "ltp": live.get("ltp"),
        "iv": g.get("impliedVolatility"),
        "delta": g.get("delta"),
        "gamma": g.get("gamma"),
        "theta": g.get("theta"),
        "vega": g.get("vega"),
        "oi": live.get("oi"),
        "volume": live.get("volume"),
        "bid": live.get("bid"),
        "ask": live.get("ask"),
    }

