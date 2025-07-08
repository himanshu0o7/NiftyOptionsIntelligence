"""
Capital Management System - Strict ₹17,000 limit enforcement
"""
import streamlit as st
from datetime import datetime
from typing import Dict, Tuple, List

class CapitalManager:
    """Manage trading capital with strict limits"""
    
    def __init__(self):
        self.max_capital = 17000  # ₹17,000 total limit
        self.max_per_trade = 3400  # ₹3,400 per position limit (20%)
        self.daily_loss_limit = 850  # ₹850 daily loss limit (5%)
        self.max_positions = 5
        
        # Lot sizes for each index
        self.lot_sizes = {
            'NIFTY': 75,
            'BANKNIFTY': 15,
            'FINNIFTY': 25,
            'MIDCPNIFTY': 50,
            'NIFTYNXT50': 120
        }
        
        # ATM ranges for each index
        self.atm_ranges = {
            'NIFTY': 200,        # ±200 points
            'BANKNIFTY': 500,    # ±500 points
            'FINNIFTY': 300,     # ±300 points
            'MIDCPNIFTY': 300,   # ±300 points
            'NIFTYNXT50': 400    # ±400 points
        }
    
    def get_current_spot_price(self, underlying: str) -> float:
        """Get current spot price for underlying"""
        # These would come from live API in production
        spot_prices = {
            'NIFTY': 23500,
            'BANKNIFTY': 50250,
            'FINNIFTY': 20150,
            'MIDCPNIFTY': 12800,
            'NIFTYNXT50': 68400
        }
        return spot_prices.get(underlying, 23500)
    
    def is_atm_strike_valid(self, strike_price: float, underlying: str) -> bool:
        """Check if strike price is ATM (within range)"""
        spot_price = self.get_current_spot_price(underlying)
        atm_range = self.atm_ranges.get(underlying, 200)
        
        return abs(strike_price - spot_price) <= atm_range
    
    def calculate_current_capital_usage(self) -> float:
        """Calculate current capital usage from active positions"""
        if 'active_positions' not in st.session_state:
            return 0.0
        
        total_used = 0.0
        for position in st.session_state.active_positions:
            if position.get('status') == 'OPEN':
                total_used += position.get('order_value', 0.0)
        
        return total_used
    
    def calculate_order_value(self, underlying: str, premium: float) -> float:
        """Calculate order value for given underlying and premium"""
        lot_size = self.lot_sizes.get(underlying, 75)
        return lot_size * premium
    
    def validate_signal_for_trading(self, signal: Dict) -> Tuple[bool, str]:
        """Comprehensive validation of signal against all limits"""
        try:
            # Extract signal details
            underlying = signal.get('underlying', 'NIFTY')
            strike_price = signal.get('strike', 0)
            premium = signal.get('premium', 200)  # Estimated premium
            
            # 1. Check ATM validation
            if not self.is_atm_strike_valid(strike_price, underlying):
                spot = self.get_current_spot_price(underlying)
                return False, f"Strike {strike_price} not ATM for {underlying} (Spot: {spot})"
            
            # 2. Check current capital usage
            current_used = self.calculate_current_capital_usage()
            if current_used >= self.max_capital:
                return False, f"Capital limit reached: ₹{current_used:,.0f} / ₹{self.max_capital:,.0f}"
            
            # 3. Calculate order value
            order_value = self.calculate_order_value(underlying, premium)
            
            # 4. Check per trade limit
            if order_value > self.max_per_trade:
                return False, f"Order value ₹{order_value:,.0f} exceeds ₹{self.max_per_trade:,.0f} limit"
            
            # 5. Check if adding this order would exceed total capital
            if (current_used + order_value) > self.max_capital:
                return False, f"Would exceed capital: ₹{current_used + order_value:,.0f} > ₹{self.max_capital:,.0f}"
            
            # 6. Check maximum positions limit
            active_positions = len([p for p in st.session_state.get('active_positions', []) if p.get('status') == 'OPEN'])
            if active_positions >= self.max_positions:
                return False, f"Maximum positions reached: {active_positions} / {self.max_positions}"
            
            return True, f"✅ Order validated: ₹{order_value:,.0f} ATM strike {strike_price}"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_atm_strike_for_underlying(self, underlying: str, option_type: str = 'CE') -> float:
        """Get ATM strike price for underlying"""
        spot_price = self.get_current_spot_price(underlying)
        
        # Round to nearest strike based on underlying
        if underlying == 'NIFTY':
            # NIFTY strikes are in multiples of 50
            atm_strike = round(spot_price / 50) * 50
        elif underlying == 'BANKNIFTY':
            # BANKNIFTY strikes are in multiples of 100
            atm_strike = round(spot_price / 100) * 100
        elif underlying in ['FINNIFTY', 'MIDCPNIFTY']:
            # FINNIFTY/MIDCPNIFTY strikes are in multiples of 50
            atm_strike = round(spot_price / 50) * 50
        elif underlying == 'NIFTYNXT50':
            # NIFTYNXT50 strikes are in multiples of 100
            atm_strike = round(spot_price / 100) * 100
        else:
            atm_strike = round(spot_price / 50) * 50
        
        return atm_strike
    
    def create_compliant_signal(self, underlying: str, action: str, confidence: float, signal_type: str) -> Dict:
        """Create a signal that complies with capital and ATM requirements"""
        option_type = 'CE' if action == 'BUY' else 'PE'
        atm_strike = self.get_atm_strike_for_underlying(underlying, option_type)
        
        # Estimate realistic premium based on underlying
        premium_estimates = {
            'NIFTY': 180,
            'BANKNIFTY': 250,
            'FINNIFTY': 150,
            'MIDCPNIFTY': 120,
            'NIFTYNXT50': 200
        }
        
        estimated_premium = premium_estimates.get(underlying, 180)
        order_value = self.calculate_order_value(underlying, estimated_premium)
        
        signal = {
            'signal_type': signal_type,
            'underlying': underlying,
            'strike': atm_strike,
            'option_type': option_type,
            'action': 'BUY',  # Only BUY orders as per requirement
            'confidence': confidence,
            'premium': estimated_premium,
            'order_value': order_value,
            'lot_size': self.lot_sizes.get(underlying, 75),
            'timestamp': datetime.now().isoformat(),
            'symbol': f"{underlying}{datetime.now().strftime('%d%b%Y').upper()}{int(atm_strike)}{option_type}",
            'reasoning': f"ATM {option_type} at {atm_strike} within ₹{self.max_per_trade:,.0f} limit"
        }
        
        return signal
    
    def get_capital_status(self) -> Dict:
        """Get current capital status"""
        current_used = self.calculate_current_capital_usage()
        available = self.max_capital - current_used
        utilization = (current_used / self.max_capital) * 100
        
        active_positions = len([p for p in st.session_state.get('active_positions', []) if p.get('status') == 'OPEN'])
        
        return {
            'total_capital': self.max_capital,
            'used_capital': current_used,
            'available_capital': available,
            'utilization_percent': utilization,
            'active_positions': active_positions,
            'max_positions': self.max_positions,
            'daily_loss_limit': self.daily_loss_limit,
            'max_per_trade': self.max_per_trade
        }
    
    def update_position_after_order(self, signal: Dict, order_id: str):
        """Update position tracking after successful order"""
        position = {
            'symbol': signal['symbol'],
            'underlying': signal['underlying'],
            'strike': signal['strike'],
            'option_type': signal['option_type'],
            'quantity': signal['lot_size'],
            'side': signal['action'],
            'entry_price': signal['premium'],
            'order_value': signal['order_value'],
            'order_id': order_id,
            'timestamp': datetime.now(),
            'status': 'OPEN'
        }
        
        if 'active_positions' not in st.session_state:
            st.session_state.active_positions = []
        
        st.session_state.active_positions.append(position)
        
        # Update capital tracking
        if 'capital_used' not in st.session_state:
            st.session_state.capital_used = 0
        
        st.session_state.capital_used += signal['order_value']