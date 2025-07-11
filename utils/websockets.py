# utils/websockets.py

from SmartApi.smartWebSocketV2 import SmartWebSocketV2
import threading
import time

sws = None  # global for control

def start_websocket_feed(tokens, token_list=None, stream_id="kp5_stream", mode=3):
    """
    Start Angel One SmartWebSocketV2 with auto-reconnect & SnapQuote mode.
    Args:
        tokens: dict with jwtToken, api_key, clientcode, feedToken
        token_list: [{"exchangeType": 2, "tokens": ["56583", "56584"]}] etc.
        stream_id: any unique string
        mode: 1=LTP, 2=Quote, 3=SnapQuote (default=3 for full data)
    """

    global sws

    if not token_list:
        # Default fallback to NIFTY-EQ in SnapQuote (exchangeType=1 = NSE)
        token_list = [{"exchangeType": 1, "tokens": ["26009"]}]

    def on_data(wsapp, message):
        symbol = message.get("tradingsymbol", message.get("token"))
        print(f"📡 [{symbol}] LTP: {message.get('last_traded_price')}, Bid: {message['best_5_buy_data'][0]['price']}, Ask: {message['best_5_sell_data'][0]['price']}")

    def on_open(wsapp):
        print("✅ WebSocket Connected. Subscribing...")
        wsapp.subscribe(stream_id, mode, token_list)

    def on_error(wsapp, error):
        print("❌ WebSocket error:", error)

    def run_ws():
        nonlocal sws
        sws = SmartWebSocketV2(
            jwt_token=tokens["jwtToken"],
            api_key=tokens["api_key"],
            client_code=tokens["clientcode"],
            feed_token=tokens["feedToken"]
        )
        sws.on_data = on_data
        sws.on_open = on_open
        sws.on_error = on_error
        try:
            sws.connect()
        except Exception as e:
            print(f"WebSocket connect failed: {e}")

    # Start main WebSocket thread
    threading.Thread(target=run_ws, daemon=True).start()

    # Auto-restart watchdog
    def watchdog():
        while True:
            time.sleep(14 * 60)  # restart every 14 minutes
            print("🔁 Auto-restarting WebSocket...")
            try:
                if sws:
                    sws.close_connection()
                time.sleep(3)
                start_websocket_feed(tokens, token_list, stream_id, mode)
            except Exception as e:
                print(f"Watchdog Error: {e}")

    threading.Thread(target=watchdog, daemon=True).start()

