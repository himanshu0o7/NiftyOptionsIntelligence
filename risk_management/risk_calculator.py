import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy.stats import norm
from dataclasses import dataclass
from utils.logger import Logger

@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    var_95: float
    var_99: float
    expected_shortfall: float
    maximum_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    volatility: float
    beta: float
    correlation: float

class RiskCalculator:
    """Advanced risk calculation and management"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = Logger()

        # Risk parameters
        self.confidence_levels = [0.95, 0.99]
        self.lookback_period = config.get('risk_lookback_days', 252)  # 1 year
        self.risk_free_rate = config.get('risk_free_rate', 0.065)  # 6.5% RBI rate

        # Risk limits
        self.max_var_limit = config.get('max_var_limit', 50000)  # Max VaR in INR
        self.max_position_size = config.get('max_position_size', 100000)  # Max position size
        self.max_portfolio_beta = config.get('max_portfolio_beta', 1.5)
        self.max_concentration = config.get('max_concentration', 0.3)  # 30%
        self.max_correlation = config.get('max_correlation', 0.8)

        self.logger.info("Risk Calculator initialized")

    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """Calculate Value at Risk"""
        try:
            if len(returns) == 0:
                return 0.0

            if method == 'historical':
                # Historical simulation method
                var = np.percentile(returns, (1 - confidence_level) * 100)

            elif method == 'parametric':
                # Parametric method (assuming normal distribution)
                mean_return = returns.mean()
                std_return = returns.std()
                var = norm.ppf(1 - confidence_level, mean_return, std_return)

            elif method == 'monte_carlo':
                # Monte Carlo simulation
                var = self._monte_carlo_var(returns, confidence_level)

            else:
                raise ValueError(f"Unknown VaR method: {method}")

            return abs(var)  # Return positive value

        except Exception as e:
            self.logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0

    def calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            if len(returns) == 0:
                return 0.0

            var = self.calculate_var(returns, confidence_level)
            # Calculate average of losses beyond VaR
            tail_losses = returns[returns <= -var]

            if len(tail_losses) == 0:
                return var

            expected_shortfall = abs(tail_losses.mean())
            return expected_shortfall

        except Exception as e:
            self.logger.error(f"Error calculating Expected Shortfall: {str(e)}")
            return 0.0

    def calculate_maximum_drawdown(self, returns: pd.Series) -> Tuple[float, int, int]:
        """Calculate Maximum Drawdown"""
        try:
            if len(returns) == 0:
                return 0.0, 0, 0

            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod()

            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()

            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max

            # Find maximum drawdown
            max_drawdown = drawdown.min()

            # Find start and end dates of maximum drawdown
            max_dd_end = drawdown.idxmin()
            max_dd_start = cumulative_returns.loc[:max_dd_end].idxmax()

            return abs(max_drawdown), max_dd_start, max_dd_end

        except Exception as e:
            self.logger.error(f"Error calculating Maximum Drawdown: {str(e)}")
            return 0.0, 0, 0

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> float:
        """Calculate Sharpe Ratio"""
        try:
            if len(returns) == 0:
                return 0.0

            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate

            # Convert annual risk-free rate to period rate
            period_risk_free_rate = risk_free_rate / 252  # Daily

            excess_returns = returns - period_risk_free_rate

            if excess_returns.std() == 0:
                return 0.0

            sharpe_ratio = excess_returns.mean() / excess_returns.std()

            # Annualize
            sharpe_ratio_annual = sharpe_ratio * np.sqrt(252)

            return sharpe_ratio_annual

        except Exception as e:
            self.logger.error(f"Error calculating Sharpe Ratio: {str(e)}")
            return 0.0

    def calculate_sortino_ratio(self, returns: pd.Series, target_return: float = 0) -> float:
        """Calculate Sortino Ratio"""
        try:
            if len(returns) == 0:
                return 0.0

            excess_returns = returns - target_return
            downside_returns = excess_returns[excess_returns < 0]

            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return float('inf') if excess_returns.mean() > 0 else 0.0

            sortino_ratio = excess_returns.mean() / downside_returns.std()

            # Annualize
            sortino_ratio_annual = sortino_ratio * np.sqrt(252)

            return sortino_ratio_annual

        except Exception as e:
            self.logger.error(f"Error calculating Sortino Ratio: {str(e)}")
            return 0.0

    def calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate Beta (sensitivity to market)"""
        try:
            if len(returns) == 0 or len(market_returns) == 0:
                return 0.0

            # Align the series
            aligned_data = pd.concat([returns, market_returns], axis=1, join='inner')

            if len(aligned_data) < 2:
                return 0.0

            portfolio_returns = aligned_data.iloc[:, 0]
            market_returns_aligned = aligned_data.iloc[:, 1]

            covariance = np.cov(portfolio_returns, market_returns_aligned)[0, 1]
            market_variance = np.var(market_returns_aligned)

            if market_variance == 0:
                return 0.0

            beta = covariance / market_variance
            return beta

        except Exception as e:
            self.logger.error(f"Error calculating Beta: {str(e)}")
            return 0.0

    def calculate_correlation(self, returns1: pd.Series, returns2: pd.Series) -> float:
        """Calculate correlation between two return series"""
        try:
            if len(returns1) == 0 or len(returns2) == 0:
                return 0.0

            correlation = returns1.corr(returns2)
            return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating correlation: {str(e)}")
            return 0.0

    def calculate_portfolio_risk_metrics(self, positions: List[Dict],
                                       market_data: Dict) -> RiskMetrics:
        """Calculate comprehensive risk metrics for portfolio"""
        try:
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(positions, market_data)

            if len(portfolio_returns) == 0:
                return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

            # Calculate market returns (using NIFTY as proxy)
            market_returns = self._get_market_returns()

            # Calculate all risk metrics
            var_95 = self.calculate_var(portfolio_returns, 0.95)
            var_99 = self.calculate_var(portfolio_returns, 0.99)
            expected_shortfall = self.calculate_expected_shortfall(portfolio_returns, 0.95)
            max_drawdown, _, _ = self.calculate_maximum_drawdown(portfolio_returns)
            sharpe_ratio = self.calculate_sharpe_ratio(portfolio_returns)
            sortino_ratio = self.calculate_sortino_ratio(portfolio_returns)
            volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            beta = self.calculate_beta(portfolio_returns, market_returns)
            correlation = self.calculate_correlation(portfolio_returns, market_returns)

            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                maximum_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                volatility=volatility,
                beta=beta,
                correlation=correlation
            )

        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk metrics: {str(e)}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

    def check_risk_limits(self, portfolio_metrics: RiskMetrics,
                         portfolio_value: float) -> List[str]:
        """Check if portfolio violates risk limits"""
        violations = []

        try:
            # VaR limit check
            var_amount = portfolio_metrics.var_95 * portfolio_value
            if var_amount > self.max_var_limit:
                violations.append(f"VaR exceeds limit: ₹{var_amount:,.0f} > ₹{self.max_var_limit:,.0f}")

            # Beta limit check
            if abs(portfolio_metrics.beta) > self.max_portfolio_beta:
                violations.append(f"Portfolio beta exceeds limit: {portfolio_metrics.beta:.2f}")

            # Volatility check (if too high)
            if portfolio_metrics.volatility > 0.5:  # 50% annual volatility
                violations.append(f"Portfolio volatility too high: {portfolio_metrics.volatility*100:.1f}%")

            # Sharpe ratio check (if too low)
            if portfolio_metrics.sharpe_ratio < -1:
                violations.append(f"Poor risk-adjusted returns: Sharpe ratio {portfolio_metrics.sharpe_ratio:.2f}")

            # Maximum drawdown check
            if portfolio_metrics.maximum_drawdown > 0.3:  # 30%
                violations.append(f"High drawdown: {portfolio_metrics.maximum_drawdown*100:.1f}%")

            return violations

        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
            return ["Error checking risk limits"]

    def calculate_position_sizing(self, expected_return: float, volatility: float,
                                 confidence_level: float = 0.95,
                                 risk_budget: float = 0.02) -> float:
        """Calculate optimal position size using Kelly Criterion and risk budgeting"""
        try:
            # Kelly Criterion
            if volatility == 0:
                return 0.0

            kelly_fraction = expected_return / (volatility ** 2)

            # Risk budgeting approach
            var_based_size = risk_budget / (norm.ppf(confidence_level) * volatility)

            # Use the more conservative of the two
            optimal_size = min(kelly_fraction, var_based_size)

            # Cap at reasonable limits
            optimal_size = max(0, min(optimal_size, 0.1))  # Max 10% of portfolio

            return optimal_size

        except Exception as e:
            self.logger.error(f"Error calculating position sizing: {str(e)}")
            return 0.02  # Default 2%

    def calculate_correlation_matrix(self, positions: List[Dict]) -> pd.DataFrame:
        """Calculate correlation matrix between positions"""
        try:
            if len(positions) < 2:
                return pd.DataFrame()

            # Get returns for each position
            returns_data = {}

            for position in positions:
                symbol = position.get('symbol', '')
                returns = self._get_symbol_returns(symbol)
                if len(returns) > 0:
                    returns_data[symbol] = returns

            if len(returns_data) < 2:
                return pd.DataFrame()

            # Create DataFrame and calculate correlation
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()

            return correlation_matrix

        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {str(e)}")
            return pd.DataFrame()

    def stress_test_portfolio(self, positions: List[Dict], scenarios: Dict[str, Dict]) -> Dict:
        """Perform stress testing on portfolio"""
        try:
            stress_results = {}

            for scenario_name, scenario_params in scenarios.items():
                market_shock = scenario_params.get('market_shock', 0)  # % change
                volatility_shock = scenario_params.get('volatility_shock', 0)  # % change

                scenario_pnl = 0

                for position in positions:
                    current_value = position.get('current_value', 0)
                    delta = position.get('delta', 0)
                    gamma = position.get('gamma', 0)
                    vega = position.get('vega', 0)

                    # Calculate P&L under stress scenario
                    delta_pnl = delta * current_value * (market_shock / 100)
                    gamma_pnl = 0.5 * gamma * current_value * ((market_shock / 100) ** 2)
                    vega_pnl = vega * (volatility_shock / 100)

                    position_pnl = delta_pnl + gamma_pnl + vega_pnl
                    scenario_pnl += position_pnl

                stress_results[scenario_name] = {
                    'total_pnl': scenario_pnl,
                    'market_shock': market_shock,
                    'volatility_shock': volatility_shock
                }

            return stress_results

        except Exception as e:
            self.logger.error(f"Error in stress testing: {str(e)}")
            return {}

    def calculate_risk_contribution(self, positions: List[Dict]) -> Dict[str, float]:
        """Calculate each position's contribution to portfolio risk"""
        try:
            risk_contributions = {}

            if len(positions) == 0:
                return risk_contributions

            # Calculate portfolio volatility
            portfolio_returns = self._calculate_portfolio_returns(positions, {})
            portfolio_vol = portfolio_returns.std() if len(portfolio_returns) > 0 else 0

            if portfolio_vol == 0:
                return {pos.get('symbol', f'pos_{i}'): 0 for i, pos in enumerate(positions)}

            # Calculate marginal risk contribution for each position
            total_value = sum(pos.get('current_value', 0) for pos in positions)

            for position in positions:
                symbol = position.get('symbol', 'unknown')
                weight = position.get('current_value', 0) / total_value if total_value > 0 else 0
                volatility = position.get('volatility', 0)

                # Simplified risk contribution calculation
                risk_contrib = weight * volatility / portfolio_vol if portfolio_vol > 0 else 0
                risk_contributions[symbol] = risk_contrib

            return risk_contributions

        except Exception as e:
            self.logger.error(f"Error calculating risk contribution: {str(e)}")
            return {}

    def _monte_carlo_var(self, returns: pd.Series, confidence_level: float,
                        simulations: int = 10000) -> float:
        """Calculate VaR using Monte Carlo simulation"""
        try:
            mean_return = returns.mean()
            std_return = returns.std()

            # Generate random returns
            simulated_returns = np.random.normal(mean_return, std_return, simulations)

            # Calculate VaR
            var = np.percentile(simulated_returns, (1 - confidence_level) * 100)

            return abs(var)

        except Exception as e:
            self.logger.error(f"Error in Monte Carlo VaR: {str(e)}")
            return 0.0

    def _calculate_portfolio_returns(self, positions: List[Dict], market_data: Dict) -> pd.Series:
        """Calculate portfolio returns time series"""
        try:
            # This is a simplified implementation
            # In practice, you'd need historical price data for all positions

            # Create dummy returns for demonstration
            # In real implementation, fetch historical data
            dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
            portfolio_returns = pd.Series(
                np.random.normal(0.001, 0.02, 252),  # 0.1% daily return, 2% volatility
                index=dates
            )

            return portfolio_returns

        except Exception as e:
            self.logger.error(f"Error calculating portfolio returns: {str(e)}")
            return pd.Series()

    def _get_market_returns(self) -> pd.Series:
        """Get market returns (NIFTY as benchmark)"""
        try:
            # This would fetch actual NIFTY returns in practice
            dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
            market_returns = pd.Series(
                np.random.normal(0.0008, 0.015, 252),  # Market returns
                index=dates
            )

            return market_returns

        except Exception as e:
            self.logger.error(f"Error getting market returns: {str(e)}")
            return pd.Series()

    def _get_symbol_returns(self, symbol: str) -> pd.Series:
        """Get returns for a specific symbol"""
        try:
            # This would fetch actual symbol returns in practice
            dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
            symbol_returns = pd.Series(
                np.random.normal(0, 0.025, 100),  # Symbol returns
                index=dates
            )

            return symbol_returns

        except Exception as e:
            self.logger.error(f"Error getting symbol returns for {symbol}: {str(e)}")
            return pd.Series()

    def generate_risk_report(self, portfolio_metrics: RiskMetrics,
                           positions: List[Dict]) -> Dict:
        """Generate comprehensive risk report"""
        try:
            risk_violations = self.check_risk_limits(portfolio_metrics,
                                                   sum(pos.get('current_value', 0) for pos in positions))

            risk_contributions = self.calculate_risk_contribution(positions)

            # Define stress test scenarios
            stress_scenarios = {
                'Market Crash': {'market_shock': -20, 'volatility_shock': 50},
                'Moderate Decline': {'market_shock': -10, 'volatility_shock': 25},
                'Volatility Spike': {'market_shock': 0, 'volatility_shock': 100},
                'Bull Rally': {'market_shock': 15, 'volatility_shock': -20}
            }

            stress_results = self.stress_test_portfolio(positions, stress_scenarios)

            correlation_matrix = self.calculate_correlation_matrix(positions)

            report = {
                'timestamp': datetime.now(),
                'risk_metrics': {
                    'var_95': portfolio_metrics.var_95,
                    'var_99': portfolio_metrics.var_99,
                    'expected_shortfall': portfolio_metrics.expected_shortfall,
                    'maximum_drawdown': portfolio_metrics.maximum_drawdown,
                    'sharpe_ratio': portfolio_metrics.sharpe_ratio,
                    'sortino_ratio': portfolio_metrics.sortino_ratio,
                    'volatility': portfolio_metrics.volatility,
                    'beta': portfolio_metrics.beta,
                    'correlation': portfolio_metrics.correlation
                },
                'risk_violations': risk_violations,
                'risk_contributions': risk_contributions,
                'stress_test_results': stress_results,
                'correlation_matrix': correlation_matrix.to_dict() if not correlation_matrix.empty else {},
                'recommendations': self._generate_risk_recommendations(portfolio_metrics, risk_violations)
            }

            return report

        except Exception as e:
            self.logger.error(f"Error generating risk report: {str(e)}")
            return {}

    def _generate_risk_recommendations(self, metrics: RiskMetrics,
                                     violations: List[str]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []

        try:
            if len(violations) > 0:
                recommendations.append("Immediate attention required: Portfolio violates risk limits")

            if metrics.sharpe_ratio < 0.5:
                recommendations.append("Consider improving risk-adjusted returns")

            if metrics.maximum_drawdown > 0.2:
                recommendations.append("Implement stricter stop-loss policies")

            if abs(metrics.beta) > 1.3:
                recommendations.append("Consider reducing market exposure")

            if metrics.volatility > 0.4:
                recommendations.append("Portfolio volatility is high - consider diversification")

            if not recommendations:
                recommendations.append("Portfolio risk profile is within acceptable limits")

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Error generating recommendations"]
