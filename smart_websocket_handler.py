"""
SmartAPI V2 WebSocket handler with automatic reconnection.

This module encapsulates the logic required to subscribe to real-time
tick data via Angel One's SmartAPI WebSocket V2. It handles
authentication using the cached session from session_manager.SessionManager,
subscribes to provided tokens, updates a shared data structure on
every tick and attempts to reconnect automatically upon errors or
disconnections.

Usage example::

    from smart_websocket_handler import SmartWebSocketHandler
    handler = SmartWebSocketHandler()
    tokens = [{"exchangeType": 2, "tokens": ["26009"]}]
    handler.start_websocket(tokens, mode=1)

mode corresponds to SmartAPI subscription modes (1=LTP, 2=Quote,
3=SnapQuote). See the SmartAPI documentation for details.
"""

import os
import threading
import time
from logzero import logger
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
from session_manager import SessionManager

# Shared in-memory store for the latest tick per token. Consumers can
# inspect this dictionary to obtain the last received LTP, open
# interest and volume values.
latest_data: dict[str, dict] = {}


class SmartWebSocketHandler:
    """Manage a SmartAPI V2 WebSocket connection with retry logic."""

    def __init__(self) -> None:
        self.sws: SmartWebSocketV2 | None = None
        self.correlation_id = "ws_handler_123"
        self.connected = False
        self.token_list: list | None = None
        self.mode: int | None = None
        self.retry_count = 0
        self.max_retries = 5
        self.session_manager = SessionManager()

    def _on_open(self, wsapp) -> None:
        """Handle WebSocket connection opened."""
        logger.info("WebSocket opened")
        self.connected = True
        self.retry_count = 0  # Reset retry count on successful connection
        
        # Subscribe to tokens once connected
        if self.token_list and self.mode is not None and self.sws:
            try:
                self.sws.subscribe(self.correlation_id, self.mode, self.token_list)
                logger.info(f"Subscribed to {len(self.token_list)} token groups")
            except Exception as e:
                logger.error(f"Failed to subscribe: {e}")

    def _on_data(self, wsapp, message) -> None:
#fix-bot-2025-07-24
        """Handle incoming tick data."""
        try:
            if isinstance(message, dict):
                # Extract token and update latest_data
                token = message.get('token')
                if token:
                    latest_data[str(token)] = message
                    logger.debug(f"Updated data for token {token}: {message}")
            else:
                logger.debug(f"Received non-dict message: {message}")
        except Exception as e:
            logger.error(f"Error processing tick data: {e}")
=======
        logger.info(f"Data: {message}")
        if isinstance(message, dict) and 'token' in message:
            token = message['token']
            latest_data[token] = {
                'ltp': message.get('last_traded_price'),
                'oi': message.get('open_interest'),
                'volume': message.get('volume_trade_for_the_day'),
                'greeks': {},
            }
 main

    def _on_error(self, wsapp, error) -> None:
        """Handle WebSocket errors."""
        logger.error(f"WebSocket error: {error}")
        self.connected = False

    def _on_close(self, wsapp) -> None:
        """Handle WebSocket connection closed."""
        logger.warning("WebSocket closed")
        self.connected = False
        
        # Attempt to reconnect if not at max retries
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            reconnect_delay = min(self.MAX_RECONNECT_DELAY, self.BACKOFF_BASE ** self.retry_count)  # Exponential backoff
            logger.info(f"Attempting reconnect {self.retry_count}/{self.max_retries} in {reconnect_delay}s")
            time.sleep(reconnect_delay)
            self._reconnect()
        else:
            logger.error("Max reconnection attempts reached. Giving up.")

    def _reconnect(self) -> None:
        """Attempt to reconnect the WebSocket."""
        try:
            # Get fresh session
            session = self.session_manager.get_session()
            if not session:
                logger.error("Failed to get fresh session for reconnection")
                return
            
            # Reinitialize WebSocket with fresh credentials
            self.sws = SmartWebSocketV2(
                session['jwtToken'],
                session['apikey'],
                session['clientcode'],
                session['feedtoken']
            )
            
            # Bind event handlers
            self.sws.on_open = self._on_open
            self.sws.on_data = self._on_data
            self.sws.on_error = self._on_error
            self.sws.on_close = self._on_close
            
            # Connect
            self.sws.connect()
            
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")

    def start_websocket(self, token_list: list, mode: int = 1) -> None:
        """Start the WebSocket connection and subscribe to tokens.
        
        Parameters
        ----------
        token_list: list
            List of token dictionaries in the format:
            [{"exchangeType": 2, "tokens": ["26009", "26017"]}]
        mode: int
            Subscription mode (1=LTP, 2=Quote, 3=SnapQuote)
        """
#fix-bot-2025-07-24
        self.token_list = token_list
        self.mode = mode
        
        try:
            # Get session credentials
            session = self.session_manager.get_session()
            if not session:
                raise RuntimeError("Failed to get session credentials")
            
            # Initialize WebSocket
            self.sws = SmartWebSocketV2(
                session['jwtToken'],
                session['apikey'],
                session['clientcode'],
                session['feedtoken']
            )
            
            # Bind event handlers
            self.sws.on_open = self._on_open
            self.sws.on_data = self._on_data
            self.sws.on_error = self._on_error
            self.sws.on_close = self._on_close
            
            # Connect in a separate thread to avoid blocking
            thread = threading.Thread(target=self.sws.connect, daemon=True)
            thread.start()
            
            logger.info("WebSocket connection initiated")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket: {e}")
            raise

    def stop_websocket(self) -> None:
        """Stop the WebSocket connection."""
        if self.sws:
            try:
                self.sws.close()
                logger.info("WebSocket stopped")
            except Exception as e:
                logger.error(f"Error stopping WebSocket: {e}")
        self.connected = False

    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.connected

    def get_latest_data(self, token: str) -> dict | None:
        """Get the latest tick data for a specific token."""
        return latest_data.get(token)

    def get_all_latest_data(self) -> dict:
        """Get all latest tick data."""
        return latest_data.copy()

        sm = SessionManager()
        session = sm.get_session()
        data = session['data']
        auth_token: str = data['jwtToken']
        client_code: str = data['clientcode']
        feed_token: str = data['feedToken']
        # Prefer API key from environment but fall back to the session data if present
        api_key: str | None = os.getenv("ANGEL_API_KEY") or data.get('api_key')
        if not api_key:
            raise EnvironmentError("ANGEL_API_KEY is not set in the environment and not available in session data")
        if not self._is_valid_api_key(api_key):
            raise ValueError("Invalid API key format or length")
        # Persist parameters for reconnection
        self.token_list = token_list
        self.mode = mode
        # Instantiate WebSocket object
        self.sws = SmartWebSocketV2(
            auth_token=auth_token,
            api_key=api_key,
            client_code=client_code,
            feed_token=feed_token,
            max_retry_attempt=10,
            retry_strategy=1,
            retry_delay=2,
            retry_multiplier=2,
            ping_interval=5,
        )
        # Attach callbacks
        self.sws.on_open = self._on_open
        self.sws.on_data = self._on_data
        self.sws.on_error = self._on_error
        self.sws.on_close = self._on_close
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

# fix-bot-2025-07-24
        def connect_thread() -> None:
            try:
                # Connect blocks until closed
                self.sws.connect()
            except Exception as exc:
                logger.error(f"Connect failed: {exc}")
                self._reconnect()

        # Run connection in a background thread to avoid blocking the caller
        threading.Thread(target=connect_thread, daemon=True).start()

    def get_latest_data(self, token: str) -> dict | None:
        """Return the last tick received for the given token."""
        return latest_data.get(token)

    def close(self) -> None:
        """Close the active WebSocket connection if one exists."""
        if self.sws:
            self.sws.close_connection()
 main
