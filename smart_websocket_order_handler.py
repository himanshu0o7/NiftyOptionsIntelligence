# smart_websocket_order_handler.py
# For order/trade updates; separate from market WS.

import threading
from logzero import logger
from SmartApi.webSocket import WebSocket  # V1-style for order WS
from session_manager import SessionManager
import time

class SmartWebSocketOrderHandler:
    def __init__(self):
        self.ws = None

    def _on_data(self, ws, message, data_type, continue_flag):
        logger.info(f"Order update: {message}")  # e.g., {'order_id': '123', 'status': 'executed'}

    def _on_error(self, ws, error):
        logger.error(f"Order WS error: {error}")

    def _on_close(self, ws):
        logger.info("Order WS closed - reconnecting...")
        time.sleep(2)
        self.ws.connect()

    def start_order_ws(self, task='ou'):  # 'ou' orders, 'tu' trades
        sm = SessionManager()
        session = sm.get_session()
        feed_token = session['data']['feedToken']
        client_code = session['data']['clientcode']

        self.ws = WebSocket(feed_token, client_code, task)
        self.ws.on_data = self._on_data
        self.ws.on_error = self._on_error
        self.ws.on_close = self._on_close

        threading.Thread(target=self.ws.connect, daemon=True).start()
        time.sleep(1)
        self.ws.subscribe(["all"])  # All orders/trades

    def close(self):
        if self.ws:
            self.ws.close()

