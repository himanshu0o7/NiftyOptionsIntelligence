"""
Professional Options Trading Strategy
Based on market sentiment and Greeks analysis for optimal strike selection
"""

from typing import Dict, List, Tuple
from datetime import datetime
import logging

class ProfessionalOptionsStrategy:
    """
    Professional options trading strategy following market sentiment rules:
    - Strong Uptrend: ATM or 1 step ITM CE
    - Slow Uptrend: ATM CE
    - Strong Downtrend: ATM or 1 step ITM PE
    - Breakdown Expected: ATM PE
    - Sideways: Avoid or wait for breakout
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Strike steps for different underlyings
        self.strike_steps = {
            'NIFTY': 50,
            'BANKNIFTY': 100,
            'FINNIFTY': 50,
            'MIDCPNIFTY': 50,
            'NIFTYNXT50': 100
        }

        # Ideal Greeks ranges for buying
        self.ideal_greeks = {
            'CE': {
                'delta': (0.45, 0.65),
                'gamma': (0.04, 0.10),
                'theta': (-1.5, -0.5),
                'vega': (0.08, 0.15)
            },
            'PE': {
                'delta': (-0.65, -0.45),
                'gamma': (0.04, 0.10),
                'theta': (-1.5, -0.5),
                'vega': (0.08, 0.15)
            }
        }

    def analyze_market_sentiment(self, underlying: str, price_data: Dict) -> str:
        """
        Analyze market sentiment using technical indicators
        Returns: 'strong_up', 'slow_up', 'sideways', 'breakdown', 'strong_down'
        """
        try:
            # These would be real technical indicators in production
            current_price = price_data.get('current_price', 0)
            rsi = price_data.get('rsi', 50)
            ema_5 = price_data.get('ema_5', current_price)
            ema_20 = price_data.get('ema_20', current_price)
            volume_ratio = price_data.get('volume_ratio', 1.0)

            # Strong uptrend conditions
            if (rsi > 60 and
                current_price > ema_5 > ema_20 and
                volume_ratio > 1.5):
                return 'strong_up'

            # Slow uptrend conditions
            elif (rsi > 50 and
                  current_price > ema_5 and
                  ema_5 > ema_20):
                return 'slow_up'

            # Strong downtrend conditions
            elif (rsi < 40 and
                  current_price < ema_5 < ema_20 and
                  volume_ratio > 1.5):
                return 'strong_down'

            # Breakdown expected
            elif (rsi < 50 and
                  current_price < ema_5 and
                  ema_5 < ema_20):
                return 'breakdown'

            # Sideways market
            else:
                return 'sideways'

        except Exception as e:
            self.logger.error(f"Market sentiment analysis error: {e}")
            return 'sideways'

    def get_optimal_strike_and_type(self, underlying: str, spot_price: float, market_sentiment: str) -> Tuple[float, str]:
        """
        Get optimal strike price and option type based on market sentiment
        """
        step = self.strike_steps.get(underlying, 50)
        atm_strike = round(spot_price / step) * step

        # CE Buy strategies
        if market_sentiment == 'strong_up':
            # Strong uptrend: ATM or 1 step ITM
            return atm_strike - step, 'CE'  # 1 step ITM

        elif market_sentiment == 'slow_up':
            # Slow uptrend: ATM
            return atm_strike, 'CE'

        # PE Buy strategies
        elif market_sentiment == 'strong_down':
            # Strong downtrend: ATM or 1 step ITM
            return atm_strike + step, 'PE'  # 1 step ITM

        elif market_sentiment == 'breakdown':
            # Breakdown expected: ATM
            return atm_strike, 'PE'

        # Sideways market - avoid trading
        else:
            return atm_strike, 'WAIT'  # Don't trade in sideways

    def validate_greeks_for_entry(self, greeks: Dict, option_type: str) -> Tuple[bool, str]:
        """
        Validate Greeks for option entry based on professional criteria
        """
        if option_type not in ['CE', 'PE']:
            return False, "Invalid option type"

        ideal = self.ideal_greeks[option_type]

        # Check Delta
        delta = greeks.get('delta', 0)
        if not (ideal['delta'][0] <= delta <= ideal['delta'][1]):
            return False, f"Delta {delta:.3f} not in ideal range {ideal['delta']}"

        # Check Gamma
        gamma = greeks.get('gamma', 0)
        if gamma < ideal['gamma'][0]:
            return False, f"Gamma {gamma:.3f} too low (need >{ideal['gamma'][0]})"

        # Check Theta
        theta = greeks.get('theta', 0)
        if theta < ideal['theta'][0]:
            return False, f"Theta {theta:.3f} too high decay (need >{ideal['theta'][0]})"

        # Check Vega
        vega = greeks.get('vega', 0)
        if not (ideal['vega'][0] <= vega <= ideal['vega'][1]):
            return False, f"Vega {vega:.3f} not in ideal range {ideal['vega']}"

        return True, "All Greeks within ideal ranges"

    def check_liquidity_and_oi(self, option_data: Dict) -> Tuple[bool, str]:
        """
        Check option liquidity and open interest
        """
        volume = option_data.get('volume', 0)
        open_interest = option_data.get('open_interest', 0)
        bid_ask_spread = option_data.get('bid_ask_spread', 0)

        # Minimum liquidity requirements
        if volume < 100:
            return False, f"Volume {volume} too low (need >100)"

        if open_interest < 1000:
            return False, f"OI {open_interest} too low (need >1000)"

        if bid_ask_spread > 2.0:
            return False, f"Bid-ask spread {bid_ask_spread} too wide (need <2.0)"

        return True, "Liquidity and OI adequate"

    def generate_professional_signal(self, underlying: str, market_data: Dict) -> Dict:
        """
        Generate professional options signal based on market sentiment and Greeks
        """
        try:
            # Analyze market sentiment
            market_sentiment = self.analyze_market_sentiment(underlying, market_data)

            # Get current price
            spot_price = market_data.get('current_price', 0)

            # Get optimal strike and option type
            optimal_strike, option_type = self.get_optimal_strike_and_type(
                underlying, spot_price, market_sentiment
            )

            # Don't trade in sideways market
            if option_type == 'WAIT':
                return {
                    'action': 'WAIT',
                    'reason': 'Sideways market - avoid option buying',
                    'market_sentiment': market_sentiment,
                    'recommendation': 'Wait for breakout or breakdown'
                }

            # Calculate expected Greeks for this strike
            moneyness = (optimal_strike - spot_price) / spot_price

            # Simulate realistic Greeks
            if option_type == 'CE':
                delta = max(0.45, 0.55 - abs(moneyness) * 2)
                gamma = max(0.04, 0.06 - abs(moneyness))
                theta = max(-1.5, -0.8 - abs(moneyness))
                vega = 0.12 + abs(moneyness) * 0.05
            else:  # PE
                delta = min(-0.45, -0.55 + abs(moneyness) * 2)
                gamma = max(0.04, 0.06 - abs(moneyness))
                theta = max(-1.5, -0.8 - abs(moneyness))
                vega = 0.12 + abs(moneyness) * 0.05

            greeks = {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            }

            # Validate Greeks
            greeks_valid, greeks_msg = self.validate_greeks_for_entry(greeks, option_type)

            # Check liquidity (simulated)
            option_data = {
                'volume': 5000,
                'open_interest': 25000,
                'bid_ask_spread': 0.5
            }
            liquidity_valid, liquidity_msg = self.check_liquidity_and_oi(option_data)

            # Generate signal
            signal = {
                'signal_type': 'Professional Options Strategy',
                'underlying': underlying,
                'action': 'BUY',
                'option_type': option_type,
                'strike': optimal_strike,
                'spot_price': spot_price,
                'market_sentiment': market_sentiment,
                'strike_type': 'ITM' if moneyness < 0 else 'ATM' if abs(moneyness) < 0.02 else 'OTM',
                'confidence': self._calculate_confidence(market_sentiment, greeks_valid, liquidity_valid),
                'greeks': greeks,
                'greeks_valid': greeks_valid,
                'greeks_message': greeks_msg,
                'liquidity_valid': liquidity_valid,
                'liquidity_message': liquidity_msg,
                'reasoning': self._generate_reasoning(market_sentiment, option_type, optimal_strike, spot_price),
                'timestamp': datetime.now().isoformat()
            }

            return signal

        except Exception as e:
            self.logger.error(f"Professional signal generation error: {e}")
            return {
                'action': 'ERROR',
                'reason': f'Signal generation failed: {str(e)}'
            }

    def _calculate_confidence(self, market_sentiment: str, greeks_valid: bool, liquidity_valid: bool) -> float:
        """Calculate confidence score based on various factors"""
        base_confidence = {
            'strong_up': 0.85,
            'slow_up': 0.75,
            'strong_down': 0.85,
            'breakdown': 0.75,
            'sideways': 0.40
        }

        confidence = base_confidence.get(market_sentiment, 0.50)

        # Adjust based on Greeks validation
        if greeks_valid:
            confidence += 0.05
        else:
            confidence -= 0.10

        # Adjust based on liquidity
        if liquidity_valid:
            confidence += 0.05
        else:
            confidence -= 0.10

        return max(0.0, min(1.0, confidence))

    def _generate_reasoning(self, market_sentiment: str, option_type: str, strike: float, spot: float) -> str:
        """Generate human-readable reasoning for the signal"""
        sentiment_desc = {
            'strong_up': 'Strong bullish momentum with high volume',
            'slow_up': 'Gradual upward trend with steady movement',
            'strong_down': 'Strong bearish momentum with high volume',
            'breakdown': 'Bearish breakdown pattern forming',
            'sideways': 'Sideways consolidation phase'
        }

        strike_desc = 'ITM' if strike != spot else 'ATM'

        reasoning = f"{sentiment_desc.get(market_sentiment, 'Market analysis')} suggests {option_type} {strike_desc} at {strike} is optimal. "

        if option_type == 'CE':
            reasoning += f"Call option benefits from upward price movement with good Greeks profile."
        else:
            reasoning += f"Put option benefits from downward price movement with good Greeks profile."

        return reasoning