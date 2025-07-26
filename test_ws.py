from smart_websocket_handler import SmartWebSocketHandler, latest_data
import time

# Token for NIFTY50 index (example token)
tokens = [{"exchangeType": 1, "tokens": ["26000"]}]  # NSE tokens

ws_handler = SmartWebSocketHandler()
ws_handler.start_websocket(tokens, mode=2)  # mode=2 means 'Quote'

# Loop to monitor incoming data
while not stop_loop:
    print("Latest Tick:", latest_data.get("26000"))
    time.sleep(2)

print("Websocket monitoring stopped.")
