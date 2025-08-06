import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from core.database import Database
from utils.logger import Logger
from indicators.greeks_calculator import GreeksCalculator

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    token: str
    quantity: int
    avg_price: float
    current_price: float
    transaction_type: str  # BUY/SELL
    product_type: str
    exchange: str
    strategy_name: str
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

class PositionManager:
    """Manage trading positions and risk"""

    def __init__(self, config: Dict, db=None):
        self.config = config
        self.logger = Logger()
        self.db = db if db else Database()
        self.greeks_calc = GreeksCalculator()

        # Position tracking
        self.active_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []

        # Risk limits
        self.max_positions = config.get('max_positions', 10)
        self.max_position_size = config.get('max_position_size', 50000)
        self.max_daily_loss = config.get('max_daily_loss', 10000)
        self.max_portfolio_delta = config.get('max_portfolio_delta', 100)
        self.max_portfolio_gamma = config.get('max_portfolio_gamma', 10)
        self.max_concentration = config.get('max_concentration', 0.3)  # 30% max in single position

        # Portfolio metrics
        self.portfolio_value = 0.0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.portfolio_greeks = {}

        self.logger.info("Position Manager initialized")

    def add_position(self, symbol: str, token: str, quantity: int, price: float,
                    transaction_type: str, product_type: str, exchange: str,
                    strategy_name: str, stop_loss: float = None,
                    take_profit: float = None) -> bool:
        """Add a new position"""
        try:
            # Check risk limits before adding position
            if not self._check_position_limits(symbol, quantity, price):
                return False

            position = Position(
                symbol=symbol,
                token=token,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                transaction_type=transaction_type,
                product_type=product_type,
                exchange=exchange,
                strategy_name=strategy_name,
                entry_time=datetime.now(),
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            # Add to active positions
            position_key = f"{symbol}_{transaction_type}"

            if position_key in self.active_positions:
                # Update existing position (average down/up)
                existing_pos = self.active_positions[position_key]
                total_quantity = existing_pos.quantity + quantity
                total_value = (existing_pos.quantity * existing_pos.avg_price) + (quantity * price)
                new_avg_price = total_value / total_quantity

                existing_pos.quantity = total_quantity
                existing_pos.avg_price = new_avg_price
                existing_pos.current_price = price

                self.logger.info(f"Updated position: {symbol}, New Qty: {total_quantity}, Avg Price: {new_avg_price}")
            else:
                self.active_positions[position_key] = position
                self.logger.info(f"Added new position: {symbol}, Qty: {quantity}, Price: {price}")

            # Save to database
            self._save_position_to_db(self.active_positions[position_key])

            # Update portfolio metrics
            self._update_portfolio_metrics()

            return True

        except Exception as e:
            self.logger.error(f"Error adding position: {str(e)}")
            return False

    def close_position(self, symbol: str, transaction_type: str,
                      exit_price: float, quantity: int = None) -> bool:
        """Close a position partially or completely"""
        try:
            position_key = f"{symbol}_{transaction_type}"

            if position_key not in self.active_positions:
                self.logger.warning(f"Position not found: {position_key}")
                return False

            position = self.active_positions[position_key]

            # Determine quantity to close
            close_quantity = quantity if quantity else position.quantity
            close_quantity = min(close_quantity, position.quantity)

            # Calculate P&L
            pnl = self._calculate_position_pnl(position, exit_price, close_quantity)

            # Update position
            position.quantity -= close_quantity
            position.realized_pnl += pnl

            # If position is fully closed, move to closed positions
            if position.quantity <= 0:
                position.quantity = 0
                position.current_price = exit_price
                self.closed_positions.append(position)
                del self.active_positions[position_key]
                self.logger.info(f"Closed position: {symbol}, P&L: {pnl}")
            else:
                self.logger.info(f"Partially closed position: {symbol}, Remaining Qty: {position.quantity}, P&L: {pnl}")

            # Update portfolio metrics
            self._update_portfolio_metrics()

            return True

        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return False

    def update_positions(self, market_data: Dict):
        """Update all positions with current market data"""
        try:
            for position_key, position in self.active_positions.items():
                symbol_data = market_data.get(position.symbol)

                if symbol_data:
                    current_price = symbol_data.get('ltp', position.current_price)
                    position.current_price = current_price

                    # Calculate unrealized P&L
                    position.unrealized_pnl = self._calculate_position_pnl(
                        position, current_price, position.quantity
                    )

                    # Update database
                    self._save_position_to_db(position)

            # Update portfolio metrics
            self._update_portfolio_metrics()

        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")

    def check_stop_loss(self, position: Position) -> bool:
        """Check if stop loss should be triggered"""
        try:
            if not position.stop_loss:
                return False

            current_price = position.current_price

            if position.transaction_type == 'BUY':
                return current_price <= position.stop_loss
            else:  # SELL
                return current_price >= position.stop_loss

        except Exception as e:
            self.logger.error(f"Error checking stop loss: {str(e)}")
            return False

    def check_take_profit(self, position: Position) -> bool:
        """Check if take profit should be triggered"""
        try:
            if not position.take_profit:
                return False

            current_price = position.current_price

            if position.transaction_type == 'BUY':
                return current_price >= position.take_profit
            else:  # SELL
                return current_price <= position.take_profit

        except Exception as e:
            self.logger.error(f"Error checking take profit: {str(e)}")
            return False

    def update_trailing_stop(self, position: Position) -> bool:
        """Update trailing stop loss"""
        try:
            if not position.trailing_stop:
                return False

            current_price = position.current_price
            trailing_pct = position.trailing_stop / 100  # Convert to decimal

            if position.transaction_type == 'BUY':
                # For long positions, trail below the price
                new_stop = current_price * (1 - trailing_pct)
                if not position.stop_loss or new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    self.logger.info(f"Updated trailing stop for {position.symbol}: {new_stop}")
                    return True
            else:  # SELL
                # For short positions, trail above the price
                new_stop = current_price * (1 + trailing_pct)
                if not position.stop_loss or new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    self.logger.info(f"Updated trailing stop for {position.symbol}: {new_stop}")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {str(e)}")
            return False

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        try:
            total_positions = len(self.active_positions)
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.closed_positions)

            # Calculate portfolio value
            portfolio_value = sum(
                abs(pos.quantity * pos.current_price) for pos in self.active_positions.values()
            )

            # Calculate portfolio Greeks for options
            portfolio_greeks = self._calculate_portfolio_greeks()

            # Get top positions by value
            position_values = [
                {
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'value': abs(pos.quantity * pos.current_price),
                    'pnl': pos.unrealized_pnl,
                    'pnl_pct': (pos.unrealized_pnl / (pos.quantity * pos.avg_price)) * 100 if pos.avg_price > 0 else 0
                }
                for pos in self.active_positions.values()
            ]

            position_values.sort(key=lambda x: x['value'], reverse=True)

            return {
                'total_positions': total_positions,
                'portfolio_value': round(portfolio_value, 2),
                'unrealized_pnl': round(total_unrealized_pnl, 2),
                'realized_pnl': round(total_realized_pnl, 2),
                'total_pnl': round(total_unrealized_pnl + total_realized_pnl, 2),
                'daily_pnl': round(self.daily_pnl, 2),
                'max_drawdown': round(self.max_drawdown, 2),
                'portfolio_greeks': portfolio_greeks,
                'top_positions': position_values[:10],
                'risk_metrics': self._calculate_risk_metrics()
            }

        except Exception as e:
            self.logger.error(f"Error generating portfolio summary: {str(e)}")
            return {}

    def get_risk_warnings(self) -> List[str]:
        """Get current risk warnings"""
        warnings = []

        try:
            # Check position count limit
            if len(self.active_positions) >= self.max_positions:
                warnings.append(f"Maximum positions limit reached: {len(self.active_positions)}/{self.max_positions}")

            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                warnings.append(f"Daily loss limit exceeded: ₹{self.daily_pnl:,.2f}")

            # Check portfolio Greeks limits
            portfolio_greeks = self._calculate_portfolio_greeks()

            if abs(portfolio_greeks.get('total_delta', 0)) > self.max_portfolio_delta:
                warnings.append(f"Portfolio delta limit exceeded: {portfolio_greeks.get('total_delta', 0)}")

            if abs(portfolio_greeks.get('total_gamma', 0)) > self.max_portfolio_gamma:
                warnings.append(f"Portfolio gamma limit exceeded: {portfolio_greeks.get('total_gamma', 0)}")

            # Check concentration risk
            portfolio_value = sum(abs(pos.quantity * pos.current_price) for pos in self.active_positions.values())

            for pos in self.active_positions.values():
                position_value = abs(pos.quantity * pos.current_price)
                concentration = position_value / portfolio_value if portfolio_value > 0 else 0

                if concentration > self.max_concentration:
                    warnings.append(f"High concentration in {pos.symbol}: {concentration*100:.1f}%")

            # Check positions nearing stop loss
            for pos in self.active_positions.values():
                if pos.stop_loss:
                    distance_to_stop = abs(pos.current_price - pos.stop_loss) / pos.current_price * 100
                    if distance_to_stop < 2:  # Within 2% of stop loss
                        warnings.append(f"{pos.symbol} near stop loss: {distance_to_stop:.1f}% away")

            return warnings

        except Exception as e:
            self.logger.error(f"Error generating risk warnings: {str(e)}")
            return ["Error calculating risk warnings"]

    def _check_position_limits(self, symbol: str, quantity: int, price: float) -> bool:
        """Check if new position violates risk limits"""
        try:
            # Check maximum positions
            if len(self.active_positions) >= self.max_positions:
                self.logger.warning("Maximum positions limit reached")
                return False

            # Check position size limit
            position_value = abs(quantity * price)
            if position_value > self.max_position_size:
                self.logger.warning(f"Position size exceeds limit: ₹{position_value:,.2f}")
                return False

            # Check concentration limit
            total_portfolio_value = sum(abs(pos.quantity * pos.current_price) for pos in self.active_positions.values())
            total_portfolio_value += position_value

            concentration = position_value / total_portfolio_value if total_portfolio_value > 0 else 0
            if concentration > self.max_concentration:
                self.logger.warning(f"Position concentration exceeds limit: {concentration*100:.1f}%")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking position limits: {str(e)}")
            return False

    def _calculate_position_pnl(self, position: Position, current_price: float, quantity: int) -> float:
        """Calculate P&L for a position"""
        try:
            if position.transaction_type == 'BUY':
                pnl = (current_price - position.avg_price) * quantity
            else:  # SELL
                pnl = (position.avg_price - current_price) * quantity

            return round(pnl, 2)

        except Exception as e:
            self.logger.error(f"Error calculating position P&L: {str(e)}")
            return 0.0

    def _calculate_portfolio_greeks(self) -> Dict[str, float]:
        """Calculate portfolio-level Greeks"""
        try:
            portfolio_greeks = {
                'total_delta': 0,
                'total_gamma': 0,
                'total_theta': 0,
                'total_vega': 0,
                'total_rho': 0
            }

            for position in self.active_positions.values():
                # Only calculate Greeks for options
                if 'CE' in position.symbol or 'PE' in position.symbol:
                    # Extract option details (simplified)
                    # In real implementation, you'd parse the symbol properly
                    spot_price = position.current_price * 100  # Approximate underlying price
                    strike_price = self._extract_strike_from_symbol(position.symbol)
                    expiry_date = self._extract_expiry_from_symbol(position.symbol)
                    option_type = 'CE' if 'CE' in position.symbol else 'PE'

                    if strike_price and expiry_date:
                        greeks = self.greeks_calc.calculate_all_greeks(
                            spot_price, strike_price, expiry_date, option_type
                        )

                        # Multiply by position size
                        portfolio_greeks['total_delta'] += greeks.get('delta', 0) * position.quantity
                        portfolio_greeks['total_gamma'] += greeks.get('gamma', 0) * position.quantity
                        portfolio_greeks['total_theta'] += greeks.get('theta', 0) * position.quantity
                        portfolio_greeks['total_vega'] += greeks.get('vega', 0) * position.quantity
                        portfolio_greeks['total_rho'] += greeks.get('rho', 0) * position.quantity

            # Round the results
            for key in portfolio_greeks:
                portfolio_greeks[key] = round(portfolio_greeks[key], 2)

            return portfolio_greeks

        except Exception as e:
            self.logger.error(f"Error calculating portfolio Greeks: {str(e)}")
            return {}

    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics"""
        try:
            positions = list(self.active_positions.values())

            if not positions:
                return {}

            # Calculate VaR (simplified)
            pnl_values = [pos.unrealized_pnl for pos in positions]
            portfolio_std = np.std(pnl_values) if len(pnl_values) > 1 else 0
            var_95 = np.percentile(pnl_values, 5) if len(pnl_values) > 1 else 0

            # Calculate Sharpe ratio (simplified daily calculation)
            returns = [pos.unrealized_pnl / (pos.quantity * pos.avg_price) for pos in positions if pos.avg_price > 0]
            avg_return = np.mean(returns) if returns else 0
            return_std = np.std(returns) if len(returns) > 1 else 0
            sharpe_ratio = (avg_return / return_std) if return_std > 0 else 0

            # Calculate maximum drawdown
            high_water_mark = max([pos.unrealized_pnl for pos in positions] + [0])
            current_value = sum(pos.unrealized_pnl for pos in positions)
            max_drawdown = (high_water_mark - current_value) / high_water_mark if high_water_mark > 0 else 0

            return {
                'portfolio_volatility': round(portfolio_std, 2),
                'var_95': round(var_95, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'max_drawdown_pct': round(max_drawdown * 100, 2),
                'position_correlation': self._calculate_position_correlation()
            }

        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}

    def _calculate_position_correlation(self) -> float:
        """Calculate average correlation between positions"""
        try:
            # Simplified correlation calculation
            # In real implementation, you'd calculate based on price movements
            return 0.5  # Placeholder

        except Exception as e:
            self.logger.error(f"Error calculating position correlation: {str(e)}")
            return 0.0

    def _extract_strike_from_symbol(self, symbol: str) -> Optional[float]:
        """Extract strike price from option symbol"""
        try:
            import re
            # Pattern to match strike price (last numbers in the symbol)
            match = re.search(r'(\d+)$', symbol)
            if match:
                return float(match.group(1))
            return None
        except:
            return None

    def _extract_expiry_from_symbol(self, symbol: str) -> Optional[str]:
        """Extract expiry date from option symbol"""
        try:
            # This is a simplified extraction
            # In real implementation, you'd parse based on Angel One symbol format
            today = datetime.now()
            # Assume weekly expiry (Thursday)
            days_ahead = (3 - today.weekday()) % 7  # Next Thursday
            if days_ahead == 0:
                days_ahead = 7
            expiry_date = today + timedelta(days=days_ahead)
            return expiry_date.strftime('%Y-%m-%d')
        except:
            return None

    def _save_position_to_db(self, position: Position):
        """Save position to database"""
        try:
            position_data = {
                'symbol': position.symbol,
                'token': position.token,
                'product_type': position.product_type,
                'exchange': position.exchange,
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'current_price': position.current_price,
                'pnl': position.realized_pnl,
                'unrealized_pnl': position.unrealized_pnl,
                'strategy_name': position.strategy_name,
                'opened_at': position.entry_time,
                'status': 'OPEN' if position.quantity > 0 else 'CLOSED'
            }

            self.db.update_position(position_data)

        except Exception as e:
            self.logger.error(f"Error saving position to database: {str(e)}")

    def _update_portfolio_metrics(self):
        """Update portfolio-level metrics"""
        try:
            # Calculate current portfolio value and P&L
            self.portfolio_value = sum(abs(pos.quantity * pos.current_price) for pos in self.active_positions.values())

            current_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            current_realized_pnl = sum(pos.realized_pnl for pos in self.closed_positions)

            self.total_pnl = current_unrealized_pnl + current_realized_pnl

            # Update daily P&L (this would be reset daily in a real system)
            self.daily_pnl = current_unrealized_pnl

            # Update max drawdown
            if self.total_pnl < self.max_drawdown:
                self.max_drawdown = self.total_pnl

            # Update portfolio Greeks
            self.portfolio_greeks = self._calculate_portfolio_greeks()

        except Exception as e:
            self.logger.error(f"Error updating portfolio metrics: {str(e)}")

    def reset_daily_metrics(self):
        """Reset daily metrics (called at start of trading day)"""
        try:
            self.daily_pnl = 0.0
            self.logger.info("Daily metrics reset")
        except Exception as e:
            self.logger.error(f"Error resetting daily metrics: {str(e)}")

    def get_positions_dataframe(self) -> pd.DataFrame:
        """Get positions as pandas DataFrame"""
        try:
            positions_data = []

            for pos in self.active_positions.values():
                positions_data.append({
                    'Symbol': pos.symbol,
                    'Quantity': pos.quantity,
                    'Avg Price': pos.avg_price,
                    'Current Price': pos.current_price,
                    'Unrealized P&L': pos.unrealized_pnl,
                    'P&L %': (pos.unrealized_pnl / (pos.quantity * pos.avg_price)) * 100 if pos.avg_price > 0 else 0,
                    'Strategy': pos.strategy_name,
                    'Entry Time': pos.entry_time,
                    'Stop Loss': pos.stop_loss,
                    'Take Profit': pos.take_profit
                })

            return pd.DataFrame(positions_data)

        except Exception as e:
            self.logger.error(f"Error creating positions DataFrame: {str(e)}")
            return pd.DataFrame()
