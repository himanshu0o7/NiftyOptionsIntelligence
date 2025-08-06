"""
Angel One WebSocket v2 implementation for live market data
"""
import websocket
import json
import threading
import time
from typing import Dict, List, Callable, Optional
from utils.logger import Logger

class AngelWebSocketV2:
    """Angel One WebSocket v2 client for real-time market data"""

    def __init__(self, auth_token: str, api_key: str, client_code: str, feed_token: str):
        self.auth_token = auth_token
        self.api_key = api_key
        self.client_code = client_code
        self.feed_token = feed_token
        self.logger = Logger()

        self.ws = None
        self.is_connected = False
        self.subscribed_tokens = []

        # Callbacks
        self.on_tick_callback = None
        self.on_error_callback = None
        self.on_connect_callback = None

        # WebSocket URL for Angel One v2
        self.ws_url = "wss://smartapisocket.angelone.in/smart-stream"

    def connect(self) -> bool:
        """Connect to Angel One WebSocket v2"""
        try:
            self.logger.info("Connecting to Angel One WebSocket v2...")

            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )

            # Start connection in separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()

            # Wait for connection
            timeout = 10
            while timeout > 0 and not self.is_connected:
                time.sleep(0.5)
                timeout -= 0.5

            if self.is_connected:
                self.logger.info("WebSocket v2 connected successfully")
                return True
            else:
                self.logger.error("WebSocket v2 connection timeout")
                return False

        except Exception as e:
            self.logger.error(f"WebSocket v2 connection error: {e}")
            return False

    def _on_open(self, ws):
        """Handle WebSocket connection open"""
        try:
            self.logger.info("WebSocket v2 connection opened")

            # Send authentication
            auth_message = {
                "a": "auth",
                "user": self.client_code,
                "token": self.feed_token
            }

            ws.send(json.dumps(auth_message))
            self.is_connected = True

            if self.on_connect_callback:
                self.on_connect_callback()

        except Exception as e:
            self.logger.error(f"WebSocket v2 open error: {e}")

    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            # Parse binary or JSON message
            if isinstance(message, bytes):
                # Binary tick data
                tick_data = self._parse_binary_tick(message)
                if tick_data and self.on_tick_callback:
                    self.on_tick_callback(tick_data)
            else:
                # JSON message
                data = json.loads(message)
                self.logger.info(f"WebSocket v2 message: {data}")

                # Handle authentication response
                if data.get('s') == 'OK':
                    self.logger.info("WebSocket v2 authenticated successfully")

        except Exception as e:
            self.logger.error(f"WebSocket v2 message error: {e}")

    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket v2 error: {error}")
        if self.on_error_callback:
            self.on_error_callback(error)

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        self.logger.info("WebSocket v2 connection closed")
        self.is_connected = False

    def subscribe(self, tokens: List[str], mode: int = 3) -> bool:
        """Subscribe to tokens for live data"""
        try:
            if not self.is_connected:
                self.logger.error("WebSocket v2 not connected")
                return False

            # Build subscription message
            subscribe_message = {
                "a": "subscribe",
                "v": tokens,
                "m": mode  # Mode: 1=LTP, 2=Quote, 3=Snap_quote
            }

            self.ws.send(json.dumps(subscribe_message))
            self.subscribed_tokens.extend(tokens)

            self.logger.info(f"Subscribed to {len(tokens)} tokens")
            return True

        except Exception as e:
            self.logger.error(f"WebSocket v2 subscription error: {e}")
            return False

    def unsubscribe(self, tokens: List[str]) -> bool:
        """Unsubscribe from tokens"""
        try:
            if not self.is_connected:
                return False

            unsubscribe_message = {
                "a": "unsubscribe",
                "v": tokens
            }

            self.ws.send(json.dumps(unsubscribe_message))

            # Remove from subscribed list
            for token in tokens:
                if token in self.subscribed_tokens:
                    self.subscribed_tokens.remove(token)

            self.logger.info(f"Unsubscribed from {len(tokens)} tokens")
            return True

        except Exception as e:
            self.logger.error(f"WebSocket v2 unsubscription error: {e}")
            return False

    def _parse_binary_tick(self, message: bytes) -> Optional[Dict]:
        """Parse binary tick data from Angel One"""
        try:
            # Angel One binary format parsing
            # This is a simplified version - actual implementation depends on Angel One's binary protocol

            if len(message) < 8:
                return None

            # Extract basic tick data (simplified)
            token = int.from_bytes(message[0:4], byteorder='big')
            ltp = int.from_bytes(message[4:8], byteorder='big') / 100.0

            tick_data = {
                'token': str(token),
                'ltp': ltp,
                'timestamp': int(time.time()),
                'exchange': 'NFO'
            }

            return tick_data

        except Exception as e:
            self.logger.error(f"Binary tick parsing error: {e}")
            return None

    def set_on_tick_callback(self, callback: Callable):
        """Set callback for tick data"""
        self.on_tick_callback = callback

    def set_on_error_callback(self, callback: Callable):
        """Set callback for errors"""
        self.on_error_callback = callback

    def set_on_connect_callback(self, callback: Callable):
        """Set callback for connection"""
        self.on_connect_callback = callback

    def disconnect(self):
        """Disconnect WebSocket"""
        try:
            if self.ws:
                self.ws.close()
            self.is_connected = False
            self.logger.info("WebSocket v2 disconnected")
        except Exception as e:
            self.logger.error(f"WebSocket v2 disconnect error: {e}")

    def get_subscribed_tokens(self) -> List[str]:
        """Get list of subscribed tokens"""
        return self.subscribed_tokens.copy()

    def is_websocket_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.is_connected