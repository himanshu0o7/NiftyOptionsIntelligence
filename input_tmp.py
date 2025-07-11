"""
Live data manager using Angel One WebSocket v2
"""
import time
from typing import Dict, List, Optional, Callable
from core.websocket_v2 import AngelWebSocketV2
from core.angel_api import AngelOneAPI
from utils.logger import Logger

class LiveDataManager:
    """Manage live market data using WebSocket v2"""
    
    def __init__(self, api_client: AngelOneAPI):
        self.api_client = api_client
        self.logger = Logger()
        self.websocket = None
        self.live_data = {}
        self.callbacks = []
        
    def connect_websocket(self) -> bool:
        """Connect to Angel One WebSocket v2"""
        try:
            if not self.api_client.is_session_valid():
                self.logger.error("API session not valid for WebSocket")
                return False
            
            # Get required tokens for WebSocket
            auth_token = self.api_client.jwt_token
            feed_token = self.api_client.getfeedToken() if hasattr(self.api_client, 'getfeedToken') else self.api_client.jwt_token
            
            self.websocket = AngelWebSocketV2(
                auth_token=auth_token,
                api_key=self.api_client.api_key,
                client_code=self.api_client.client_code,
                feed_token=feed_token
            )
            
            # Set callbacks
            self.websocket.set_on_tick_callback(self._on_tick_data)
            self.websocket.set_on_error_callback(self._on_websocket_error)
            self.websocket.set_on_connect_callback(self._on_websocket_connect)
            
            # Connect
            if self.websocket.connect():
                self.logger.info("WebSocket v2 connected successfully")
                return True
            else:
                self.logger.error("WebSocket v2 connection failed")
                return False
                
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
            return False
    
    def subscribe_to_options(self, option_tokens: List[str]) -> bool:
        """Subscribe to option tokens for live data"""
        try:
            if not self.websocket or not self.websocket.is_websocket_connected():
                self.logger.error("WebSocket not connected")
                return False
            
            # Subscribe with snap quote mode (mode 3)
            success = self.websocket.subscribe(option_tokens, mode=3)
            
            if success:
                self.logger.info(f"Subscribed to {len(option_tokens)} option tokens")
                return True
            else:
                self.logger.error("Failed to subscribe to option tokens")
                return False
                
        except Exception as e:
            self.logger.error(f"Subscription error: {e}")
            return False
    
    def get_live_option_data(self, token: str) -> Optional[Dict]:
        """Get live data for specific option token"""
        try:
            return self.live_data.get(token)
        except Exception as e:
            self.logger.error(f"Error getting live data: {e}")
            return None
    
    def get_option_premium(self, token: str) -> Optional[float]:
        """Get current premium for option"""
        try:
            data = self.live_data.get(token)
            if data:
                return data.get('ltp', 0)
            return None
        except Exception as e:
            self.logger.error(f"Error getting premium: {e}")
            return None
    
    def validate_order_amount(self, token: str, lot_size: int, max_amount: float = 17000) -> bool:
        """Validate if order amount is within limits"""
        try:
            premium = self.get_option_premium(token)
            if premium:
                order_value = premium * lot_size
                if order_value <= max_amount:
                    self.logger.info(f"Order value ₹{order_value:,.0f} within limit of ₹{max_amount:,.0f}")
                    return True
                else:
                    self.logger.error(f"Order value ₹{order_value:,.0f} exceeds limit of ₹{max_amount:,.0f}")
                    return False
            else:
                self.logger.error("Premium data not available")
                return False
        except Exception as e:
            self.logger.error(f"Error validating order amount: {e}")
            return False
    
    def _on_tick_data(self, tick_data: Dict):
        """Handle incoming tick data"""
        try:
            token = tick_data.get('token')
            if token:
                # Update live data
                self.live_data[token] = {
                    'ltp': tick_data.get('ltp', 0),
                    'volume': tick_data.get('volume', 0),
                    'oi': tick_data.get('oi', 0),
                    'timestamp': int(time.time()),
                    'bid': tick_data.get('bid', 0),
                    'ask': tick_data.get('ask', 0),
                    'change': tick_data.get('change', 0)
                }
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(token, self.live_data[token])
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
                        
        except Exception as e:
            self.logger.error(f"Tick data processing error: {e}")
    
    def _on_websocket_error(self, error):
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket error: {error}")
    
    def _on_websocket_connect(self):
        """Handle WebSocket connection"""
        self.logger.info("WebSocket connected and ready for subscriptions")
    
    def add_data_callback(self, callback: Callable):
        """Add callback for live data updates"""
        self.callbacks.append(callback)
    
    def disconnect(self):
        """Disconnect WebSocket"""
        try:
            if self.websocket:
                self.websocket.disconnect()
            self.logger.info("Live data manager disconnected")
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")
    
    def get_market_status(self) -> Dict:
        """Get current market status"""
        return {
            'websocket_connected': self.websocket.is_websocket_connected() if self.websocket else False,
            'subscribed_tokens': len(self.websocket.get_subscribed_tokens()) if self.websocket else 0,
            'live_data_count': len(self.live_data),
            'last_update': max([data.get('timestamp', 0) for data in self.live_data.values()]) if self.live_data else 0
        }