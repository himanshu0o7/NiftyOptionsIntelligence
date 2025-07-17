# smart_websocket_v1_handler.py
# V1 alternative for stability; no changes to V2 module.

import threading
from logzero import logger  # pip install logzero if needed; fallback to print
from SmartApi.webSocket import WebSocket  # V1 import from SDK
from session_manager import SessionManager
import time

# Global data store (matches your V2)
latest_data = {}

class SmartWebSocketV1Handler:
    def __init__(self):
        self.ws = None
        self.connected = False

    def _on_open(self, ws):
        logger.info("V1 WebSocket opened")
        self.connected = True

    def _on_data(self, ws, message, data_type, continue_flag):
        logger.info(f"V1 data received: {message}")
        if isinstance(message, dict) and 'token' in message:
            token = message['token']
            latest_data[token] = {
                'ltp': message.get('ltp'),
                'oi': message.get('oi'),
                'volume': message.get('vtt'),  # V1 field for volume
                'greeks': {}  # External compute
            }

    def _on_error(self, ws, error):
        logger.error(f"V1 error: {error}")
        self.connected = False
        self._reconnect()

    def _on_close(self, ws):
        logger.info("V1 closed")
        self.connected = False
        self._reconnect()

    def _reconnect(self):
        time.sleep(2)  # Simple delay
        logger.warning("V1 reconnecting...")
        self.start_websocket(self.token_list, self.mode)  # Re-init

    def start_websocket(self, token_list, mode='QUOTE'):  # V1 modes: LTP/QUOTE/SNAPQUOTE
        sm = SessionManager()
        session = sm.get_session()
        data = session['data']
        feed_token = data['feedToken']
        client_code = data['clientcode']

        time.sleep(1)  # Rate delay

        self.token_list = token_list
        self.mode = mode
        self.ws = WebSocket(feed_token, client_code, mode)
        self.ws.on_open = self._on_open
        self.ws.on_data = self._on_data
        self.ws.on_error = self._on_error
        self.ws.on_close = self._on_close

        def connect_and_subscribe():
            try:
                self.ws.connect()
                time.sleep(1)
                if self.connected:
                    # V1 subscribe: strings like "nfo|token"
                    tokens = [f"nfo|{t}" for item in token_list for t in item['tokens']]
                    self.ws.subscribe(tokens)
            except Exception as e:
                logger.error(f"V1 subscribe failed: {e}")

        threading.Thread(target=connect_and_subscribe, daemon=True).start()

    def get_latest_data(self, token):
        return latest_data.get(token)

    def close(self):
        if self.ws:
            self.ws.close()

# Example usage (test standalone):
# handler = SmartWebSocketV1Handler()
# token_list = [{"exchangeType": 2, "tokens": ["26009"]}]
# handler.start_websocket(token_list, mode='QUOTE')

