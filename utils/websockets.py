# utils/websockets.py

from SmartApi.smartWebSocketV2 import SmartWebSocketV2
import threading
import time

sws = None  # Global reference

def start_websocket_feed(tokens, token_list=None, stream_id="kp5_stream"):
    """
    Starts Angel One WebSocket with auto-reconnect every 15 minutes.
    """
    global sws

    if not token_list:
        # Example: NIFTY-EQ token (or customize CE/PE later)
        token_list = [{"exchangeType": 1, "tokens": ["26009"]}]

    def on_data(wsapp, message):
        print("📡 Live Tick:", message)

    def on_open(wsapp):
        print("✅ WebSocket opened. Subscribing...")
        wsapp.subscribe(stream_id, 1, token_list)  # mode 1 = LTP

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
            print(f"WebSocket connection failed: {e}")

    # Start in background thread
    threading.Thread(target=run_ws, daemon=True).start()

    # Auto-restart thread every 14 minutes
    def watchdog():
        while True:
            time.sleep(14 * 60)
            print("🔄 Restarting WebSocket...")
            try:
                sws.close_connection()
                time.sleep(2)
                start_websocket_feed(tokens, token_list, stream_id)
            except Exception as e:
                print(f"Error in auto-restart: {e}")

    threading.Thread(target=watchdog, daemon=True).start()

