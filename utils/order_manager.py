"""
Order manager using official Angel One SmartAPI format
"""
from typing import Dict, Optional
from core.angel_api import AngelOneAPI
from core.symbol_resolver import SymbolResolver
from utils.logger import Logger

class OrderManager:
    """Manage order placement using proper Angel One format"""
    
    def __init__(self, api_client: AngelOneAPI):
        self.api_client = api_client
        self.symbol_resolver = SymbolResolver()
        self.logger = Logger()
        
    def place_option_order(self, signal: Dict) -> Optional[str]:
        """Place option order using Angel One SmartAPI format"""
        try:
            # Validate signal has required fields
            required_fields = ['symbol', 'token', 'action', 'lot_size', 'option_type']
            if not all(field in signal for field in required_fields):
                self.logger.error(f"Signal missing required fields: {signal}")
                return None
            
            # Validate symbol and token
            if not self.symbol_resolver.validate_symbol_token(signal['symbol'], signal['token']):
                self.logger.error(f"Invalid symbol/token: {signal['symbol']}/{signal['token']}")
                # Try to resolve again
                new_symbol, new_token = self.symbol_resolver.get_option_symbol_and_token(
                    signal['underlying'], signal['expiry'], signal['strike'], signal['option_type']
                )
                if new_symbol and new_token:
                    signal['symbol'] = new_symbol
                    signal['token'] = new_token
                    self.logger.info(f"Resolved to: {new_symbol}/{new_token}")
                else:
                    return None
            
            # Calculate proper quantity (must be multiple of lot size)
            lot_size = signal.get('lot_size', 50)  # Default NIFTY lot size
            quantity = lot_size  # Always trade 1 lot as per requirement
            
            # Build order parameters in Angel One format
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": signal['symbol'],
                "symboltoken": signal['token'],
                "transactiontype": signal['action'],  # "BUY" or "SELL"
                "exchange": "NFO",
                "ordertype": "MARKET",  # Market order for immediate execution
                "producttype": "CARRYFORWARD",  # For overnight positions
                "duration": "DAY",
                "price": "0",  # Market order
                "squareoff": "0",
                "stoploss": "0",
                "quantity": str(quantity)  # Proper lot size multiple
            }
            
            self.logger.info(f"Placing order: {order_params['tradingsymbol']} {order_params['transactiontype']} {order_params['quantity']}")
            
            # Place order using Angel One API
            order_id = self.api_client.place_order(order_params)
            
            if order_id:
                self.logger.info(f"Order placed successfully: {order_id}")
                return order_id
            else:
                self.logger.error("Order placement failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def place_test_order(self, underlying: str = "NIFTY") -> Optional[str]:
        """Place a test order to verify system works"""
        try:
            # Create test signal
            spot_prices = {'NIFTY': 23500, 'BANKNIFTY': 51000}
            spot_price = spot_prices.get(underlying, 23500)
            strike = round(spot_price / 50) * 50  # ATM strike
            
            expiry_dates = {'NIFTY': '10JUL25', 'BANKNIFTY': '09JUL25'}
            expiry = expiry_dates.get(underlying, '10JUL25')
            
            # Get symbol and token
            symbol, token = self.symbol_resolver.get_option_symbol_and_token(
                underlying, expiry, strike, 'CE'
            )
            
            if not symbol or not token:
                self.logger.error(f"Could not resolve test option for {underlying}")
                return None
            
            test_signal = {
                'symbol': symbol,
                'token': token,
                'strike': strike,
                'expiry': expiry,
                'option_type': 'CE',
                'lot_size': 50 if underlying == 'NIFTY' else 15,
                'action': 'BUY',
                'underlying': underlying
            }
            
            return self.place_option_order(test_signal)
            
        except Exception as e:
            self.logger.error(f"Error placing test order: {e}")
            return None
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status from Angel One"""
        try:
            order_book = self.api_client.get_order_book()
            if order_book:
                for order in order_book:
                    if order.get('orderid') == order_id:
                        return order
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            return self.api_client.cancel_order(order_id)
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False