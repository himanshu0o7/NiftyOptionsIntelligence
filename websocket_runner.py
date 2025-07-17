# websocket_runner.py - Updated for SmartAPI WebSocket V2

from SmartApi.smartWebSocketV2 import WebSocketV2
import os
from dotenv import load_dotenv

load_dotenv()

# Load credentials from .env
FEED_TOKEN = os.getenv("FEED_TOKEN")
CLIENT_CODE = os.getenv("CLIENT_ID")

# Symbol tokens (use NSE option tokens e.g. 26000 for NIFTY)
# Get correct tokens from SmartAPI instrument dump or SnapQuote
TOKEN_LIST = ["26000"]  # Replace with actual symbolToken(s)

# Initialize WebSocketV2
sws = WebSocketV2(
    feed_token=FEED_TOKEN,
    client_code=CLIENT_CODE,
    script_tokens=TOKEN_LIST
)

# Define callbacks
def on_open(wsapp):
    print("✅ WebSocket V2 connection opened")
    sws.subscribe(TOKEN_LIST)

def on_data(wsapp, message):
    print(f"📡 Tick Data: {message}")

def on_error(wsapp, error):
    print(f"❌ WebSocket Error: {error}")

def on_close(wsapp):
    print("🔌 WebSocket connection closed")

# Bind callbacks
sws.on_open = on_open
sws.on_data = on_data
sws.on_error = on_error
sws.on_close = on_close

# Start WebSocket (safely without signal handler crash)
from twisted.internet import reactor
reactor.run(installSignalHandlers=False)

