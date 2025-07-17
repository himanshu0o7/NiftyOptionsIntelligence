# smart_websocket_handler.py
# Fixed with V2 docs: higher retries, ping_interval, subscribe in on_open, resubscribe in on_close.

import threading
from logzero import logger
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from session_manager import SessionManager
reactor.run(installSignalHandlers=False)


import time


latest_data = {}

class SmartWebSocketHandler:
    def __init__(self):
        self.sws = None
        self.correlation_id = "ws_handler_123"
        self.connected = False
        self.token_list = None  # New: Store for resubscribe
        self.mode = None  # New: Store for resubscribe

    def _on_open(self, wsapp):
        logger.info("WebSocket opened")
        self.connected = True
        if self.token_list and self.mode:  # Subscribe here (doc-recommended)
            self.sws.subscribe(self.correlation_id, self.mode, self.token_list)

    def _on_data(self, wsapp, message):
        logger.info(f"Data: {message}")
        if isinstance(message, dict) and 'token' in message:
            token = message['token']
            latest_data[token] = {
                'ltp': message.get('last_traded_price'),
                'oi': message.get('open_interest'),
                'volume': message.get('volume_trade_for_the_day'),
                'greeks': {}
            }

    def _on_error(self, wsapp, error):
        logger.error(f"Error: {error}")
        self.connected = False
        self._reconnect()  # Trigger reconnect

    def _on_close(self, wsapp):
        logger.info("Closed")
        self.connected = False
        self._reconnect()  # Doc: Resubscribe after reconnect

    def _reconnect(self):
        time.sleep(2)  # Initial delay
        logger.warning("Reconnecting...")
        self.start_websocket(self.token_list, self.mode)  # Re-init with stored params

    def start_websocket(self, token_list, mode=2):
        sm = SessionManager()
        session = sm.get_session()
        data = session['data']
        auth_token = data['jwtToken']
        api_key = 'your_api_key'  # From env/config
        client_code = data['clientcode']
        feed_token = data['feedToken']

        time.sleep(1)  # Rate delay

        self.token_list = token_list
        self.mode = mode
        self.sws = SmartWebSocketV2(
            auth_token=auth_token,  # Doc: Keyword OK
            api_key=api_key,
            client_code=client_code,
            feed_token=feed_token,
            max_retry_attempt=10,  # Doc: Increase for stability
            retry_strategy=1,  # Exponential backoff
            retry_delay=2,
            retry_multiplier=2,
            ping_interval=5  # Doc: Heartbeat to prevent idle close
        )
        self.sws.on_open = self._on_open
        self.sws.on_data = self._on_data
        self.sws.on_error = self._on_error
        self.sws.on_close = self._on_close

        def connect_thread():
            try:
                self.sws.connect()
            except Exception as e:
                logger.error(f"Connect failed: {e}")
                self._reconnect()

        threading.Thread(target=connect_thread, daemon=True).start()

    def get_latest_data(self, token):
        return latest_data.get(token)

    def close(self):
        if self.sws:
            self.sws.close_connection()

