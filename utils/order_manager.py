"""
Order manager using official Angel One SmartAPI format
"""
from typing import Dict, Optional
from core.angel_api import AngelOneAPI
from core.symbol_resolver import SymbolResolver
from utils.logger import Logger
from risk_management.audit_filters import AuditBasedFilters
from utils.telegram_notifier import TelegramNotifier

class OrderManager:
    """Manage order placement using proper Angel One format"""
    
    def __init__(self, api_client: AngelOneAPI):
        self.api_client = api_client
        self.symbol_resolver = SymbolResolver()
        self.logger = Logger()
        self.audit_filters = AuditBasedFilters()
        self.telegram = TelegramNotifier()
        
    def place_option_order(self, signal: Dict) -> Optional[str]:
        """Place option order with comprehensive validation"""
        try:
            # Validate signal has required fields
            required_fields = ['symbol', 'token', 'action', 'lot_size', 'option_type']
            if not all(field in signal for field in required_fields):
                self.logger.error(f"Signal missing required fields: {signal}")
                return None
            
            # 1. Capital Check - Ensure order value <= 17,000
            lot_size = signal.get('lot_size', 75)
            premium = signal.get('premium', 0)
            
            if premium > 0:
                order_value = premium * lot_size
                if order_value > 17000:
                    self.logger.error(f"Order value ₹{order_value:,.0f} exceeds capital limit of ₹17,000")
                    return None
                self.logger.info(f"Order value: ₹{order_value:,.0f} (within ₹17,000 limit)")
            
            # 2. Greeks Validation
            required_greeks = ['delta', 'gamma', 'theta', 'vega', 'implied_volatility']
            greeks_valid = all(signal.get(greek) is not None for greek in required_greeks)
            
            if not greeks_valid:
                self.logger.error("Missing Greeks data - order rejected")
                return None
            
            # Check Greeks thresholds using audit filters
            if self.audit_filters.check_greeks_based_sl(signal):
                self.logger.error("Greeks-based SL criteria triggered - order rejected")
                return None
            
            delta = abs(signal.get('delta', 0))
            iv = signal.get('implied_volatility', 0)
            
            if delta < 0.1:
                self.logger.error(f"Delta too low: {delta:.3f} - order rejected")
                return None
            
            if iv < 10 or iv > 50:
                self.logger.error(f"IV out of range: {iv:.1f}% - order rejected")
                return None
            
            # 3. Volume and OI Validation
            volume = signal.get('trade_volume', 0)
            oi_change = signal.get('oi_change', 0)
            
            if volume < 1000:
                self.logger.error(f"Insufficient volume: {volume:,.0f} - order rejected")
                return None
            
            if abs(oi_change) < 100:
                self.logger.error(f"Insufficient OI change: {oi_change:,.0f} - order rejected")
                return None
            
            # 4. Technical Indicators Check
            confidence = signal.get('confidence', 0)
            liquidity_score = signal.get('liquidity_score', 0)
            
            if confidence < 0.7:
                self.logger.error(f"Confidence too low: {confidence:.1%} - order rejected")
                return None
            
            if liquidity_score < 0.6:
                self.logger.error(f"Liquidity too low: {liquidity_score:.1f} - order rejected")
                return None
            
            # 5. Validate symbol and token
            if not self.symbol_resolver.validate_symbol_token(signal['symbol'], signal['token']):
                self.logger.error(f"Invalid symbol/token: {signal['symbol']}/{signal['token']}")
                return None
            
            # 6. Calculate quantity (exactly 1 lot)
            quantity = lot_size  # Always 1 lot only
            
            # 7. Final validation - ensure quantity is multiple of lot size
            if quantity % lot_size != 0:
                self.logger.error(f"Quantity {quantity} not multiple of lot size {lot_size}")
                return None
            
            # Build order parameters
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": signal['symbol'],
                "symboltoken": signal['token'],
                "transactiontype": signal['action'],
                "exchange": "NFO",
                "ordertype": "MARKET",
                "producttype": "CARRYFORWARD",
                "duration": "DAY",
                "price": "0",
                "squareoff": "0",
                "stoploss": "0",
                "quantity": str(quantity)
            }
            
            self.logger.info(f"All validations passed - placing order: {signal['symbol']} quantity {quantity}")
            self.logger.info(f"Greeks: Δ{delta:.3f} IV{iv:.1f}% Volume{volume:,.0f} OI±{oi_change:,.0f}")
            
            # Place order using Angel One API
            order_id = self.api_client.place_order(order_params)
            
            if order_id:
                self.logger.info(f"Order placed successfully: {order_id}")
                return order_id
            else:
                self.logger.error("Order placement failed at API level")
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
                'lot_size': 75 if underlying == 'NIFTY' else 35,
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