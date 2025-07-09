"""
Single Trade Strategy - One trade at a time with ₹17k max capital per trade
"""
import streamlit as st
from datetime import datetime, time
from typing import Dict, Optional
import pandas as pd

class SingleTradeStrategy:
    """Strategy for single intraday option trade with ₹17k max capital"""
    
    def __init__(self):
        self.max_capital_per_trade = 17000  # ₹17k max per single trade
        self.lot_quantity = 1  # Only 1 lot per trade
        
        # Lot sizes for each index
        self.lot_sizes = {
            'NIFTY': 75,
            'BANKNIFTY': 15,
            'FINNIFTY': 25,
            'MIDCPNIFTY': 50,
            'NIFTYNXT50': 120
        }
    
    def can_place_new_trade(self) -> bool:
        """Check if we can place a new trade (no active positions)"""
        active_positions = st.session_state.get('active_positions', [])
        open_positions = [p for p in active_positions if p.get('status') == 'OPEN']
        
        # Only allow new trade if no active positions
        return len(open_positions) == 0
    
    def is_market_hours(self) -> bool:
        """Check if market is open (9:15 AM to 3:30 PM)"""
        now = datetime.now()
        current_time = now.time()
        
        market_open = time(9, 15)  # 9:15 AM
        market_close = time(15, 30)  # 3:30 PM
        
        # Check if it's a weekday and within market hours
        is_weekday = now.weekday() < 5  # Monday=0, Sunday=6
        return is_weekday and market_open <= current_time <= market_close
    
    def validate_trade_capital(self, underlying: str, premium: float) -> tuple[bool, str]:
        """Validate if trade is within ₹17k capital limit"""
        lot_size = self.lot_sizes.get(underlying, 75)
        trade_value = lot_size * premium
        
        if trade_value > self.max_capital_per_trade:
            return False, f"Trade value ₹{trade_value:,.0f} exceeds ₹17k limit"
        
        return True, f"Trade value ₹{trade_value:,.0f} within ₹17k limit"
    
    def generate_single_trade_signal(self, underlying: str = 'NIFTY') -> Optional[Dict]:
        """Generate signal for single intraday trade"""
        
        # Check if we can place new trade
        if not self.can_place_new_trade():
            return None
        
        # Check market hours
        if not self.is_market_hours():
            return None
        
        # Get current spot price (mock for now)
        spot_prices = {
            'NIFTY': 23500,
            'BANKNIFTY': 50250,
            'FINNIFTY': 20150,
            'MIDCPNIFTY': 12800,
            'NIFTYNXT50': 68400
        }
        
        spot_price = spot_prices.get(underlying, 23500)
        
        # Select ATM strike (rounded to nearest 50 for NIFTY)
        if underlying == 'NIFTY':
            atm_strike = round(spot_price / 50) * 50
        elif underlying == 'BANKNIFTY':
            atm_strike = round(spot_price / 100) * 100
        else:
            atm_strike = round(spot_price / 50) * 50
        
        # Simple market trend detection (mock - replace with real analysis)
        trend = self.detect_market_trend(underlying)
        
        # Choose option type based on trend
        if trend == 'BULLISH':
            option_type = 'CE'
        elif trend == 'BEARISH':
            option_type = 'PE'
        else:
            option_type = 'CE'  # Default to CE for neutral
        
        # Estimate premium (mock - replace with live data)
        estimated_premium = self.estimate_option_premium(underlying, atm_strike, option_type)
        
        # Validate capital requirement
        is_valid, message = self.validate_trade_capital(underlying, estimated_premium)
        
        if not is_valid:
            return None
        
        # Create signal
        signal = {
            'underlying': underlying,
            'symbol': f"{underlying}{atm_strike}{option_type}",
            'strike': atm_strike,
            'option_type': option_type,
            'action': 'BUY',
            'lot_size': self.lot_sizes[underlying],
            'quantity': 1,  # 1 lot only
            'premium': estimated_premium,
            'order_value': self.lot_sizes[underlying] * estimated_premium,
            'confidence': 0.75,  # 75% confidence
            'strategy': 'SINGLE_TRADE',
            'trade_type': 'INTRADAY',
            'timestamp': datetime.now(),
            'market_trend': trend,
            'capital_validation': message,
            'stop_loss': estimated_premium * 0.15,  # 15% SL
            'target': estimated_premium * 1.25,   # 25% target
            'reasoning': f"Single intraday {option_type} trade on {underlying} trend: {trend}"
        }
        
        return signal
    
    def detect_market_trend(self, underlying: str) -> str:
        """Simple trend detection (replace with real technical analysis)"""
        # Mock trend detection - replace with actual indicators
        import random
        trends = ['BULLISH', 'BEARISH', 'NEUTRAL']
        return random.choice(trends)
    
    def estimate_option_premium(self, underlying: str, strike: float, option_type: str) -> float:
        """Estimate option premium (replace with live data)"""
        # Mock premium estimation - replace with actual market data
        base_premiums = {
            'NIFTY': 180,
            'BANKNIFTY': 220,
            'FINNIFTY': 160,
            'MIDCPNIFTY': 140,
            'NIFTYNXT50': 95
        }
        
        base_premium = base_premiums.get(underlying, 180)
        
        # Add some randomness for realistic simulation
        import random
        premium = base_premium + random.randint(-30, 50)
        
        return max(premium, 50)  # Minimum ₹50 premium
    
    def get_strategy_status(self) -> Dict:
        """Get current strategy status"""
        active_positions = st.session_state.get('active_positions', [])
        open_positions = [p for p in active_positions if p.get('status') == 'OPEN']
        
        if open_positions:
            current_trade = open_positions[0]
            return {
                'status': 'TRADE_ACTIVE',
                'message': f"Active trade: {current_trade.get('symbol', 'Unknown')}",
                'can_trade': False,
                'current_trade': current_trade
            }
        else:
            return {
                'status': 'READY_FOR_TRADE',
                'message': 'Ready for new intraday trade',
                'can_trade': self.is_market_hours(),
                'current_trade': None
            }
    
    def close_intraday_positions(self):
        """Close all intraday positions at 3:25 PM"""
        now = datetime.now()
        close_time = time(15, 25)  # 3:25 PM
        
        if now.time() >= close_time:
            active_positions = st.session_state.get('active_positions', [])
            for position in active_positions:
                if position.get('status') == 'OPEN':
                    # Mark position for closure
                    position['status'] = 'CLOSING'
                    position['close_reason'] = 'INTRADAY_AUTO_CLOSE'
                    position['close_time'] = now
            
            st.session_state.active_positions = active_positions
            return True
        
        return False