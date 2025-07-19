# utils/websockets.py
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
import threading


from SmartApi.smartWebSocketV2 import SmartWebSocketV2  # Third-party import
def start_websocket_feed(tokens, token_list=None, on_data=None, on_open=None, on_error=None):
    def ws_connect():
        sws = SmartWebSocketV2(
            jwt_token=tokens["jwtToken"],
            api_key=tokens["api_key"],
            client_code=tokens["clientcode"],
            feed_token=tokens["feedToken"]
        )
        # Attach user callbacks
        if on_data: sws.on_data = on_data
        if on_open: sws.on_open = on_open
        if on_error: sws.on_error = on_error

        try:
            sws.connect()
            # Subscribe logic (pass your token_list, mode, correlation_id here)
            # Example:
            # sws.subscribe("kp5correlation", 3, token_list)
        except Exception as e:
            print(f"WebSocket Error: {e}")
            if "401" in str(e) or "Invalid Feed Token" in str(e):
                print("🔁 Token expired. Please refresh tokens and restart WebSocket.")

    # Run WebSocket in background thread
    threading.Thread(target=ws_connect, daemon=True).start()

# Fix functions with missing docstrings and clean up code
def example_function():
    """Example function docstring."""
    pass
