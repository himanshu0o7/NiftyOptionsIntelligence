"""
Audit-based risk management filters
Implementation of audit recommendations for automated risk controls
"""
import yaml
from typing import Dict, Optional
from datetime import datetime, timedelta
from utils.logger import Logger

class AuditBasedFilters:
    """Implement audit-based risk management filters"""

    def __init__(self, config_path: str = "risk_config.yaml"):
        self.logger = Logger()
        self.config = self._load_config(config_path)
        self.mtm_loss_tracker = 0
        self.trading_halted = False

    def _load_config(self, config_path: str) -> Dict:
        """Load risk configuration from YAML"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading risk config: {e}")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Default risk configuration if file not found"""
        return {
            'risk_management': {
                'total_capital': 17000,
                'max_daily_loss': 850,
                'stop_loss': {
                    'greeks_based_sl': {'delta_threshold': 0.05}
                },
                'trailing_stop_loss': {
                    'activation_profit': 15,
                    'trail_percent': 10
                }
            },
            'alerts': {
                'mtm_alerts': {'loss_threshold_percent': 2}
            }
        }

    def check_greeks_based_sl(self, position: Dict) -> bool:
        """Check if position should be squared off based on Greeks"""
        try:
            delta = abs(position.get('delta', 0))
            iv = position.get('implied_volatility', 0)

            delta_threshold = self.config['risk_management']['stop_loss']['greeks_based_sl']['delta_threshold']

            if delta < delta_threshold:
                self.logger.warning(f"Greeks-based SL triggered: Delta {delta:.3f} < {delta_threshold}")
                return True

            if iv < 5:  # IV too low
                self.logger.warning(f"Greeks-based SL triggered: IV {iv:.1f}% too low")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error in Greeks SL check: {e}")
            return False

    def check_trailing_stop_activation(self, position: Dict, current_price: float) -> Dict:
        """Check and update trailing stop loss"""
        try:
            entry_price = position.get('entry_price', 0)
            highest_price = position.get('highest_price', entry_price)

            if entry_price <= 0:
                return position

            # Calculate current profit percentage
            profit_pct = (current_price - entry_price) / entry_price * 100
            activation_threshold = self.config['risk_management']['trailing_stop_loss']['activation_profit']

            # Activate TSL if profit >= 15%
            if profit_pct >= activation_threshold and not position.get('tsl_active'):
                position['tsl_active'] = True
                position['highest_price'] = current_price
                self.logger.info(f"TSL activated at {profit_pct:.1f}% profit")

            # Update trailing stop if active
            if position.get('tsl_active'):
                if current_price > highest_price:
                    position['highest_price'] = current_price

                trail_percent = self.config['risk_management']['trailing_stop_loss']['trail_percent']
                trailing_stop_price = highest_price * (1 - trail_percent / 100)

                position['trailing_stop_price'] = trailing_stop_price

                # Check if TSL should trigger
                if current_price <= trailing_stop_price:
                    position['tsl_triggered'] = True
                    self.logger.warning(f"TSL triggered: Price {current_price} <= Stop {trailing_stop_price}")

            return position

        except Exception as e:
            self.logger.error(f"Error in TSL check: {e}")
            return position

    def check_mtm_loss_alert(self, current_mtm_loss: float) -> Dict:
        """Check MTM loss and trigger alerts/halt trading"""
        try:
            capital = self.config['risk_management']['total_capital']
            loss_threshold_pct = self.config['alerts']['mtm_alerts']['loss_threshold_percent']
            halt_threshold_pct = self.config['risk_management']['max_daily_loss'] / capital * 100

            loss_percentage = (current_mtm_loss / capital) * 100

            result = {
                'alert_triggered': False,
                'halt_trading': False,
                'message': '',
                'loss_percentage': loss_percentage
            }

            # Check if loss threshold reached (2% of capital)
            if loss_percentage >= loss_threshold_pct and not self.trading_halted:
                result['alert_triggered'] = True
                result['message'] = f"MTM Loss Alert: {loss_percentage:.1f}% of capital (₹{current_mtm_loss:,.0f})"
                self.logger.warning(result['message'])

            # Check if trading should be halted (5% of capital)
            if loss_percentage >= halt_threshold_pct:
                result['halt_trading'] = True
                result['message'] = f"Trading Halted: Daily loss limit reached {loss_percentage:.1f}% (₹{current_mtm_loss:,.0f})"
                self.trading_halted = True
                self.logger.error(result['message'])

            self.mtm_loss_tracker = current_mtm_loss
            return result

        except Exception as e:
            self.logger.error(f"Error in MTM loss check: {e}")
            return {'alert_triggered': False, 'halt_trading': False, 'message': '', 'loss_percentage': 0}

    def check_take_profit_rules(self, position: Dict, current_price: float) -> bool:
        """Check if take profit should be triggered"""
        try:
            entry_price = position.get('entry_price', 0)
            entry_time = position.get('entry_time')

            if entry_price <= 0:
                return False

            profit_pct = (current_price - entry_price) / entry_price * 100

            # Quick profit rule: 20% in first 30 minutes
            if entry_time:
                time_diff = datetime.now() - datetime.fromisoformat(entry_time)
                if time_diff < timedelta(minutes=30) and profit_pct >= 20:
                    self.logger.info(f"Quick profit TP triggered: {profit_pct:.1f}% in {time_diff}")
                    return True

            # Target profit amount
            profit_amount = (current_price - entry_price) * position.get('quantity', 0)
            target_profit = self.config['risk_management']['take_profit']['target_profit_amount']

            if profit_amount >= target_profit:
                self.logger.info(f"Target profit TP triggered: ₹{profit_amount:,.0f}")
                return True

            # End of day profit (after 3 PM)
            current_time = datetime.now().time()
            if current_time.hour >= 15 and profit_pct >= 50:
                self.logger.info(f"EOD profit TP triggered: {profit_pct:.1f}%")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error in take profit check: {e}")
            return False

    def get_market_mode(self, nifty_trend: str, volatility: float) -> str:
        """Determine market mode based on trend and volatility"""
        try:
            if volatility > 25:  # High volatility
                return "rangebound"
            elif nifty_trend.upper() == "BULLISH":
                return "bullish"
            elif nifty_trend.upper() == "BEARISH":
                return "bearish"
            else:
                return "rangebound"

        except Exception as e:
            self.logger.error(f"Error determining market mode: {e}")
            return "rangebound"

    def validate_position_risk(self, position: Dict) -> Dict:
        """Comprehensive position risk validation"""
        try:
            current_price = position.get('current_price', 0)

            result = {
                'greeks_sl_triggered': self.check_greeks_based_sl(position),
                'take_profit_triggered': self.check_take_profit_rules(position, current_price),
                'position_updated': position,
                'action_required': None
            }

            # Update TSL
            result['position_updated'] = self.check_trailing_stop_activation(position, current_price)

            # Determine action required
            if result['greeks_sl_triggered']:
                result['action_required'] = "SQUARE_OFF_GREEKS_SL"
            elif result['take_profit_triggered']:
                result['action_required'] = "SQUARE_OFF_TAKE_PROFIT"
            elif result['position_updated'].get('tsl_triggered'):
                result['action_required'] = "SQUARE_OFF_TSL"

            return result

        except Exception as e:
            self.logger.error(f"Error in position risk validation: {e}")
            return {
                'greeks_sl_triggered': False,
                'take_profit_triggered': False,
                'position_updated': position,
                'action_required': None
            }

    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed based on current risk status"""
        return not self.trading_halted

    def reset_daily_tracking(self):
        """Reset daily tracking (call at market open)"""
        self.mtm_loss_tracker = 0
        self.trading_halted = False
        self.logger.info("Daily risk tracking reset")

    def get_risk_summary(self) -> Dict:
        """Get current risk management summary"""
        return {
            'total_capital': self.config['risk_management']['total_capital'],
            'current_mtm_loss': self.mtm_loss_tracker,
            'trading_halted': self.trading_halted,
            'daily_loss_limit': self.config['risk_management']['max_daily_loss'],
            'risk_per_trade': self.config['risk_management']['risk_per_trade']
        }