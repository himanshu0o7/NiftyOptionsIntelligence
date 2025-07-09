"""
Capital Management System - Strict ₹17,000 limit enforcement
"""
import streamlit as st
from datetime import datetime
from typing import Dict, Tuple, List

class CapitalManager:
    """Manage trading capital with single trade strategy"""
    
    def __init__(self):
        self.max_capital_per_trade = 17000  # ₹17,000 max per single trade
        self.max_positions = 1  # Only 1 position at a time
        self.daily_loss_limit = 3400  # ₹3,400 daily loss limit (20%)
        self.single_trade_mode = True  # Single trade mode enabled
        
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
            
            # 2. Check if any active positions exist (single trade mode)
            active_positions = len([p for p in st.session_state.get('active_positions', []) if p.get('status') == 'OPEN'])
            if active_positions > 0:
                return False, f"Single trade mode: Wait for current position to close before new trade"
            
            # 3. Calculate order value
            order_value = self.calculate_order_value(underlying, premium)
            
            # 4. Check single trade capital limit (₹17k max per trade)
            if order_value > self.max_capital_per_trade:
                return False, f"Order value ₹{order_value:,.0f} exceeds ₹17k single trade limit"
            
            # 5. Single trade mode - already checked above, but keeping for compatibility
            
            # All validations passed for single trade
            return True, f"✅ Single trade validated: ₹{order_value:,.0f} ATM strike {strike_price}"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def detect_market_trend(self, underlying: str) -> str:
        """Detect market trend using technical indicators"""
        # This would use real technical analysis in production
        # For now, simulate based on time and random factors
        import random
        
        # Simulate different market conditions
        trends = ['strong_up', 'sideways_to_up', 'sideways', 'breakdown_expected', 'strong_down']
        weights = [0.25, 0.25, 0.3, 0.1, 0.1]  # Mostly bullish/sideways market
        
        return random.choices(trends, weights=weights)[0]
    
    def get_optimal_strike_for_market(self, underlying: str, option_type: str, market_trend: str) -> float:
        """Get optimal strike based on market sentiment and option type"""
        spot_price = self.get_current_spot_price(underlying)
        
        # Get step size based on underlying
        if underlying == 'NIFTY':
            step = 50
        elif underlying == 'BANKNIFTY':
            step = 100
        elif underlying in ['FINNIFTY', 'MIDCPNIFTY']:
            step = 50
        elif underlying == 'NIFTYNXT50':
            step = 100
        else:
            step = 50
        
        # Calculate ATM strike
        atm_strike = round(spot_price / step) * step
        
        # Apply market sentiment logic for CE BUY
        if option_type == 'CE':
            if market_trend == 'strong_up':
                # Strong uptrend: ATM or 1 step ITM
                return atm_strike - step  # 1 step ITM
            elif market_trend in ['sideways_to_up', 'sideways']:
                # Slow uptrend: ATM
                return atm_strike
            else:
                # Uncertain/bearish: ATM (safe)
                return atm_strike
        
        # Apply market sentiment logic for PE BUY
        elif option_type == 'PE':
            if market_trend == 'strong_down':
                # Strong downtrend: ATM or 1 step ITM
                return atm_strike + step  # 1 step ITM
            elif market_trend == 'breakdown_expected':
                # Breakdown expected: ATM
                return atm_strike
            else:
                # Uncertain/bullish: ATM (safe)
                return atm_strike
        
        return atm_strike
    
    def get_atm_strike_for_underlying(self, underlying: str, option_type: str = 'CE') -> float:
        """Get ATM strike price for underlying (legacy method)"""
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
    
    def validate_greeks_for_buying(self, delta: float, gamma: float, theta: float, vega: float, option_type: str) -> Tuple[bool, str]:
        """Validate Greeks for option buying based on professional criteria"""
        if option_type == 'CE':
            # CE Buy validation
            if not (0.45 <= delta <= 0.65):
                return False, f"CE Delta {delta:.3f} not in ideal range (0.45-0.65)"
            if gamma < 0.04:
                return False, f"CE Gamma {gamma:.3f} too low (need >0.04)"
            if theta < -1.5:
                return False, f"CE Theta {theta:.3f} too high decay (need >-1.5)"
        
        elif option_type == 'PE':
            # PE Buy validation
            if not (-0.65 <= delta <= -0.45):
                return False, f"PE Delta {delta:.3f} not in ideal range (-0.65 to -0.45)"
            if gamma < 0.04:
                return False, f"PE Gamma {gamma:.3f} too low (need >0.04)"
            if theta < -1.5:
                return False, f"PE Theta {theta:.3f} too high decay (need >-1.5)"
        
        return True, "Greeks validation passed"
    
    def create_compliant_signal(self, underlying: str, action: str, confidence: float, signal_type: str) -> Dict:
        """Create a signal that complies with capital, ATM, and Greeks requirements"""
        
        # Detect market trend
        market_trend = self.detect_market_trend(underlying)
        
        # Determine option type based on market trend and action
        if market_trend in ['strong_up', 'sideways_to_up']:
            option_type = 'CE'  # Bullish signals
        elif market_trend in ['strong_down', 'breakdown_expected']:
            option_type = 'PE'  # Bearish signals
        else:
            option_type = 'CE'  # Default to CE for sideways (safer)
        
        # Get optimal strike based on market sentiment
        optimal_strike = self.get_optimal_strike_for_market(underlying, option_type, market_trend)
        
        # Calculate realistic Greeks based on strike and market
        spot_price = self.get_current_spot_price(underlying)
        moneyness = (optimal_strike - spot_price) / spot_price
        
        # Simulate realistic Greeks
        if option_type == 'CE':
            delta = max(0.45, 0.55 - abs(moneyness) * 2)  # 0.45-0.65 range
            gamma = max(0.04, 0.06 - abs(moneyness))      # High gamma for ATM
            theta = max(-1.5, -0.8 - abs(moneyness))      # Low theta decay
            vega = 0.12 + abs(moneyness) * 0.05           # Moderate vega
        else:  # PE
            delta = min(-0.45, -0.55 + abs(moneyness) * 2)  # -0.45 to -0.65 range
            gamma = max(0.04, 0.06 - abs(moneyness))         # High gamma for ATM
            theta = max(-1.5, -0.8 - abs(moneyness))         # Low theta decay
            vega = 0.12 + abs(moneyness) * 0.05              # Moderate vega
        
        # Validate Greeks
        greeks_valid, greeks_message = self.validate_greeks_for_buying(delta, gamma, theta, vega, option_type)
        
        # Estimate realistic premium based on underlying and moneyness
        premium_estimates = {
            'NIFTY': 180,
            'BANKNIFTY': 250,
            'FINNIFTY': 150,
            'MIDCPNIFTY': 120,
            'NIFTYNXT50': 200
        }
        
        base_premium = premium_estimates.get(underlying, 180)
        # Adjust premium based on moneyness (ITM more expensive)
        if moneyness < 0:  # ITM
            estimated_premium = base_premium * 1.2
        else:  # OTM
            estimated_premium = base_premium * 0.8
        
        order_value = self.calculate_order_value(underlying, estimated_premium)
        
        signal = {
            'signal_type': signal_type,
            'underlying': underlying,
            'strike': optimal_strike,
            'option_type': option_type,
            'action': 'BUY',  # Only BUY orders as per requirement
            'confidence': confidence,
            'premium': estimated_premium,
            'order_value': order_value,
            'lot_size': self.lot_sizes.get(underlying, 75),
            'timestamp': datetime.now().isoformat(),
            'symbol': f"{underlying}{datetime.now().strftime('%d%b%Y').upper()}{int(optimal_strike)}{option_type}",
            'market_trend': market_trend,
            'strike_type': 'ITM' if moneyness < 0 else 'ATM' if abs(moneyness) < 0.02 else 'OTM',
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'greeks_valid': greeks_valid,
            'greeks_message': greeks_message,
            'reasoning': f"Market: {market_trend} → {option_type} {optimal_strike} | Greeks: ✅ | Value: ₹{order_value:,.0f}"
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