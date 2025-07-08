import websocket
import json
import threading
import time
from typing import Dict, List, Callable, Optional
import struct
import base64
from utils.logger import Logger

class WebSocketClient:
    """Angel One WebSocket client for real-time market data"""
    
    def __init__(self, api_client):
        self.api_client = api_client
        self.logger = Logger()
        
        self.ws_url = "wss://smartapisocket.angelone.in/smart-stream"
        self.ws = None
        self.is_connected = False
        self.subscribed_tokens = set()
        
        # Callbacks
        self.on_tick_callback = None
        self.on_order_update_callback = None
        self.on_error_callback = None
        
        # Heartbeat
        self.heartbeat_interval = 30
        self.last_heartbeat = time.time()
        
    def connect(self) -> bool:
        """Connect to WebSocket server"""
        try:
            # Enable debug mode
            websocket.enableTrace(False)
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                header={
                    'Authorization': f'Bearer {self.api_client.jwt_token}',
                    'x-api-key': self.api_client.api_key,
                    'x-client-code': self.api_client.client_code,
                    'x-feed-token': self.api_client.feed_token
                },
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket in a separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection
            timeout = 10
            while not self.is_connected and timeout > 0:
                time.sleep(0.1)
                timeout -= 0.1
            
            if self.is_connected:
                self.logger.info("WebSocket connected successfully")
                # Start heartbeat
                self._start_heartbeat()
                return True
            else:
                self.logger.error("WebSocket connection timeout")
                return False
                
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from WebSocket server"""
        try:
            if self.ws:
                self.is_connected = False
                self.ws.close()
                self.logger.info("WebSocket disconnected")
        except Exception as e:
            self.logger.error(f"WebSocket disconnect error: {str(e)}")
    
    def _on_open(self, ws):
        """WebSocket connection opened"""
        self.is_connected = True
        self.logger.info("WebSocket connection opened")
        
        # Send authentication message
        auth_message = {
            "a": "auth",
            "user": self.api_client.client_code,
            "token": self.api_client.feed_token
        }
        
        self.ws.send(json.dumps(auth_message))
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            # Update last heartbeat
            self.last_heartbeat = time.time()
            
            # Parse binary message
            if isinstance(message, bytes):
                self._parse_binary_message(message)
            else:
                # JSON message
                data = json.loads(message)
                self._handle_json_message(data)
                
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {str(e)}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket error: {str(error)}")
        if self.on_error_callback:
            self.on_error_callback(error)
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        self.is_connected = False
        self.logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
    
    def _parse_binary_message(self, message: bytes):
        """Parse binary market data message"""
        try:
            # Angel One WebSocket sends binary data in specific format
            # This is a simplified parser - actual implementation would be more complex
            
            if len(message) < 8:
                return
            
            # Extract token (first 4 bytes)
            token = struct.unpack('>I', message[0:4])[0]
            
            # Extract timestamp (next 4 bytes)
            timestamp = struct.unpack('>I', message[4:8])[0]
            
            # Extract price data (remaining bytes)
            if len(message) >= 16:
                ltp = struct.unpack('>I', message[8:12])[0] / 100.0  # Divide by 100 for price
                volume = struct.unpack('>I', message[12:16])[0]
                
                tick_data = {
                    'token': str(token),
                    'timestamp': timestamp,
                    'ltp': ltp,
                    'volume': volume,
                    'last_traded_time': int(time.time())
                }
                
                if self.on_tick_callback:
                    self.on_tick_callback(tick_data)
                    
        except Exception as e:
            self.logger.error(f"Error parsing binary message: {str(e)}")
    
    def _handle_json_message(self, data: Dict):
        """Handle JSON messages from WebSocket"""
        try:
            message_type = data.get('t', '')
            
            if message_type == 'ck':
                # Connection acknowledgment
                self.logger.info("WebSocket authentication successful")
            
            elif message_type == 'tk':
                # Tick data
                if self.on_tick_callback:
                    self.on_tick_callback(data)
            
            elif message_type == 'ou':
                # Order update
                if self.on_order_update_callback:
                    self.on_order_update_callback(data)
            
            elif message_type == 'hb':
                # Heartbeat
                self.logger.debug("Received heartbeat")
                
        except Exception as e:
            self.logger.error(f"Error handling JSON message: {str(e)}")
    
    def subscribe(self, tokens: List[str], mode: str = "QUOTE"):
        """Subscribe to market data for given tokens"""
        try:
            if not self.is_connected:
                self.logger.error("WebSocket not connected")
                return False
            
            # Prepare subscription message
            subscription_message = {
                "a": "subscribe",
                "v": tokens,
                "m": mode.lower()  # quote, ltp, snap_quote, depth
            }
            
            self.ws.send(json.dumps(subscription_message))
            
            # Add tokens to subscribed set
            self.subscribed_tokens.update(tokens)
            
            self.logger.info(f"Subscribed to {len(tokens)} tokens in {mode} mode")
            return True
            
        except Exception as e:
            self.logger.error(f"Error subscribing to tokens: {str(e)}")
            return False
    
    def unsubscribe(self, tokens: List[str]):
        """Unsubscribe from market data for given tokens"""
        try:
            if not self.is_connected:
                self.logger.error("WebSocket not connected")
                return False
            
            # Prepare unsubscription message
            unsubscription_message = {
                "a": "unsubscribe",
                "v": tokens
            }
            
            self.ws.send(json.dumps(unsubscription_message))
            
            # Remove tokens from subscribed set
            self.subscribed_tokens.difference_update(tokens)
            
            self.logger.info(f"Unsubscribed from {len(tokens)} tokens")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unsubscribing from tokens: {str(e)}")
            return False
    
    def set_on_tick_callback(self, callback: Callable):
        """Set callback for tick data"""
        self.on_tick_callback = callback
    
    def set_on_order_update_callback(self, callback: Callable):
        """Set callback for order updates"""
        self.on_order_update_callback = callback
    
    def set_on_error_callback(self, callback: Callable):
        """Set callback for errors"""
        self.on_error_callback = callback
    
    def _start_heartbeat(self):
        """Start heartbeat thread"""
        def heartbeat():
            while self.is_connected:
                try:
                    current_time = time.time()
                    if current_time - self.last_heartbeat > self.heartbeat_interval * 2:
                        # No message received for too long, reconnect
                        self.logger.warning("Heartbeat timeout, reconnecting...")
                        self.disconnect()
                        time.sleep(5)
                        self.connect()
                    
                    time.sleep(self.heartbeat_interval)
                    
                except Exception as e:
                    self.logger.error(f"Heartbeat error: {str(e)}")
                    break
        
        heartbeat_thread = threading.Thread(target=heartbeat)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
    
    def get_subscribed_tokens(self) -> List[str]:
        """Get list of currently subscribed tokens"""
        return list(self.subscribed_tokens)
    
    def is_websocket_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.is_connected
