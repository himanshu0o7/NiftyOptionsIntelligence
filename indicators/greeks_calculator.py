import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, date
from typing import Dict, Optional, Tuple, List
from utils.logger import Logger

class GreeksCalculator:
    """Calculate option Greeks using Black-Scholes model"""
    
    def __init__(self):
        self.logger = Logger()
        self.risk_free_rate = 0.065  # Default RBI repo rate (6.5%)
    
    def set_risk_free_rate(self, rate: float):
        """Set the risk-free rate for calculations"""
        self.risk_free_rate = rate
    
    def calculate_time_to_expiry(self, expiry_date: str) -> float:
        """Calculate time to expiry in years"""
        try:
            if isinstance(expiry_date, str):
                expiry = datetime.strptime(expiry_date, '%Y-%m-%d').date()
            else:
                expiry = expiry_date
            
            today = date.today()
            days_to_expiry = (expiry - today).days
            
            # Convert to years (assuming 365 days in a year)
            time_to_expiry = max(days_to_expiry / 365.0, 0.0001)  # Minimum 0.0001 to avoid division by zero
            
            return time_to_expiry
            
        except Exception as e:
            self.logger.error(f"Error calculating time to expiry: {str(e)}")
            return 0.0001
    
    def black_scholes_call(self, spot: float, strike: float, time_to_expiry: float,
                          risk_free_rate: float, volatility: float) -> float:
        """Calculate call option price using Black-Scholes formula"""
        try:
            if time_to_expiry <= 0 or volatility <= 0:
                return max(spot - strike, 0)  # Intrinsic value
            
            d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            d2 = d1 - volatility * np.sqrt(time_to_expiry)
            
            call_price = (spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
            
            return max(call_price, 0)
            
        except Exception as e:
            self.logger.error(f"Error calculating Black-Scholes call price: {str(e)}")
            return 0
    
    def black_scholes_put(self, spot: float, strike: float, time_to_expiry: float,
                         risk_free_rate: float, volatility: float) -> float:
        """Calculate put option price using Black-Scholes formula"""
        try:
            if time_to_expiry <= 0 or volatility <= 0:
                return max(strike - spot, 0)  # Intrinsic value
            
            d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            d2 = d1 - volatility * np.sqrt(time_to_expiry)
            
            put_price = (strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1))
            
            return max(put_price, 0)
            
        except Exception as e:
            self.logger.error(f"Error calculating Black-Scholes put price: {str(e)}")
            return 0
    
    def calculate_delta(self, spot: float, strike: float, time_to_expiry: float,
                       risk_free_rate: float, volatility: float, option_type: str) -> float:
        """Calculate Delta (price sensitivity to underlying price change)"""
        try:
            if time_to_expiry <= 0:
                if option_type.upper() == 'CALL' or option_type.upper() == 'CE':
                    return 1.0 if spot > strike else 0.0
                else:
                    return -1.0 if spot < strike else 0.0
            
            d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            
            if option_type.upper() == 'CALL' or option_type.upper() == 'CE':
                delta = norm.cdf(d1)
            else:  # PUT or PE
                delta = norm.cdf(d1) - 1
            
            return delta
            
        except Exception as e:
            self.logger.error(f"Error calculating Delta: {str(e)}")
            return 0
    
    def calculate_gamma(self, spot: float, strike: float, time_to_expiry: float,
                       risk_free_rate: float, volatility: float) -> float:
        """Calculate Gamma (rate of change of Delta)"""
        try:
            if time_to_expiry <= 0 or volatility <= 0:
                return 0
            
            d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            
            gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(time_to_expiry))
            
            return gamma
            
        except Exception as e:
            self.logger.error(f"Error calculating Gamma: {str(e)}")
            return 0
    
    def calculate_theta(self, spot: float, strike: float, time_to_expiry: float,
                       risk_free_rate: float, volatility: float, option_type: str) -> float:
        """Calculate Theta (time decay)"""
        try:
            if time_to_expiry <= 0:
                return 0
            
            d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            d2 = d1 - volatility * np.sqrt(time_to_expiry)
            
            if option_type.upper() == 'CALL' or option_type.upper() == 'CE':
                theta = (-spot * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) -
                        risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
            else:  # PUT or PE
                theta = (-spot * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) +
                        risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2))
            
            # Convert to daily theta (divide by 365)
            theta_daily = theta / 365
            
            return theta_daily
            
        except Exception as e:
            self.logger.error(f"Error calculating Theta: {str(e)}")
            return 0
    
    def calculate_vega(self, spot: float, strike: float, time_to_expiry: float,
                      risk_free_rate: float, volatility: float) -> float:
        """Calculate Vega (sensitivity to volatility changes)"""
        try:
            if time_to_expiry <= 0:
                return 0
            
            d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            
            vega = spot * norm.pdf(d1) * np.sqrt(time_to_expiry)
            
            # Convert to vega per 1% change in volatility
            vega_percent = vega / 100
            
            return vega_percent
            
        except Exception as e:
            self.logger.error(f"Error calculating Vega: {str(e)}")
            return 0
    
    def calculate_rho(self, spot: float, strike: float, time_to_expiry: float,
                     risk_free_rate: float, volatility: float, option_type: str) -> float:
        """Calculate Rho (sensitivity to interest rate changes)"""
        try:
            if time_to_expiry <= 0:
                return 0
            
            d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
            d2 = d1 - volatility * np.sqrt(time_to_expiry)
            
            if option_type.upper() == 'CALL' or option_type.upper() == 'CE':
                rho = strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
            else:  # PUT or PE
                rho = -strike * time_to_expiry * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)
            
            # Convert to rho per 1% change in interest rate
            rho_percent = rho / 100
            
            return rho_percent
            
        except Exception as e:
            self.logger.error(f"Error calculating Rho: {str(e)}")
            return 0
    
    def calculate_implied_volatility(self, option_price: float, spot: float, strike: float,
                                   time_to_expiry: float, risk_free_rate: float,
                                   option_type: str, max_iterations: int = 100) -> float:
        """Calculate implied volatility using Newton-Raphson method"""
        try:
            if time_to_expiry <= 0:
                return 0
            
            # Initial guess for volatility
            volatility = 0.2
            
            for i in range(max_iterations):
                if option_type.upper() == 'CALL' or option_type.upper() == 'CE':
                    bs_price = self.black_scholes_call(spot, strike, time_to_expiry, risk_free_rate, volatility)
                else:
                    bs_price = self.black_scholes_put(spot, strike, time_to_expiry, risk_free_rate, volatility)
                
                # Calculate vega for Newton-Raphson method
                vega = self.calculate_vega(spot, strike, time_to_expiry, risk_free_rate, volatility)
                
                if abs(vega) < 1e-6:  # Avoid division by zero
                    break
                
                # Newton-Raphson iteration
                volatility_new = volatility - (bs_price - option_price) / (vega * 100)  # vega is per 1%
                
                if abs(volatility_new - volatility) < 1e-6:  # Convergence check
                    break
                
                volatility = max(volatility_new, 0.001)  # Ensure positive volatility
            
            return max(volatility, 0.001)
            
        except Exception as e:
            self.logger.error(f"Error calculating implied volatility: {str(e)}")
            return 0.2  # Default volatility
    
    def calculate_all_greeks(self, spot: float, strike: float, expiry_date: str,
                           option_type: str, option_price: float = None,
                           volatility: float = None) -> Dict[str, float]:
        """Calculate all Greeks for an option"""
        try:
            time_to_expiry = self.calculate_time_to_expiry(expiry_date)
            
            # Calculate implied volatility if not provided
            if volatility is None and option_price is not None:
                volatility = self.calculate_implied_volatility(
                    option_price, spot, strike, time_to_expiry, self.risk_free_rate, option_type
                )
            elif volatility is None:
                volatility = 0.2  # Default volatility
            
            # Calculate theoretical price
            if option_type.upper() == 'CALL' or option_type.upper() == 'CE':
                theoretical_price = self.black_scholes_call(spot, strike, time_to_expiry, self.risk_free_rate, volatility)
            else:
                theoretical_price = self.black_scholes_put(spot, strike, time_to_expiry, self.risk_free_rate, volatility)
            
            # Calculate all Greeks
            delta = self.calculate_delta(spot, strike, time_to_expiry, self.risk_free_rate, volatility, option_type)
            gamma = self.calculate_gamma(spot, strike, time_to_expiry, self.risk_free_rate, volatility)
            theta = self.calculate_theta(spot, strike, time_to_expiry, self.risk_free_rate, volatility, option_type)
            vega = self.calculate_vega(spot, strike, time_to_expiry, self.risk_free_rate, volatility)
            rho = self.calculate_rho(spot, strike, time_to_expiry, self.risk_free_rate, volatility, option_type)
            
            # Calculate intrinsic and time value
            if option_type.upper() == 'CALL' or option_type.upper() == 'CE':
                intrinsic_value = max(spot - strike, 0)
            else:
                intrinsic_value = max(strike - spot, 0)
            
            time_value = theoretical_price - intrinsic_value
            
            return {
                'theoretical_price': round(theoretical_price, 2),
                'intrinsic_value': round(intrinsic_value, 2),
                'time_value': round(time_value, 2),
                'implied_volatility': round(volatility * 100, 2),  # Convert to percentage
                'delta': round(delta, 4),
                'gamma': round(gamma, 6),
                'theta': round(theta, 2),
                'vega': round(vega, 2),
                'rho': round(rho, 4),
                'time_to_expiry': round(time_to_expiry, 4),
                'moneyness': round(spot / strike, 4)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Greeks: {str(e)}")
            return {}
    
    def calculate_portfolio_greeks(self, positions: List[Dict]) -> Dict[str, float]:
        """Calculate portfolio-level Greeks"""
        try:
            portfolio_greeks = {
                'total_delta': 0,
                'total_gamma': 0,
                'total_theta': 0,
                'total_vega': 0,
                'total_rho': 0,
                'net_premium': 0
            }
            
            for position in positions:
                quantity = position.get('quantity', 0)
                greeks = position.get('greeks', {})
                premium = position.get('premium', 0)
                
                # Multiply Greeks by position size
                portfolio_greeks['total_delta'] += greeks.get('delta', 0) * quantity
                portfolio_greeks['total_gamma'] += greeks.get('gamma', 0) * quantity
                portfolio_greeks['total_theta'] += greeks.get('theta', 0) * quantity
                portfolio_greeks['total_vega'] += greeks.get('vega', 0) * quantity
                portfolio_greeks['total_rho'] += greeks.get('rho', 0) * quantity
                portfolio_greeks['net_premium'] += premium * quantity
            
            # Round the results
            for key in portfolio_greeks:
                portfolio_greeks[key] = round(portfolio_greeks[key], 2)
            
            return portfolio_greeks
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio Greeks: {str(e)}")
            return {}
    
    def get_option_classification(self, spot: float, strike: float, option_type: str) -> str:
        """Classify option as ITM, ATM, or OTM"""
        try:
            moneyness_ratio = spot / strike
            
            if option_type.upper() == 'CALL' or option_type.upper() == 'CE':
                if moneyness_ratio > 1.02:
                    return 'ITM'  # In-the-money
                elif moneyness_ratio < 0.98:
                    return 'OTM'  # Out-of-the-money
                else:
                    return 'ATM'  # At-the-money
            else:  # PUT or PE
                if moneyness_ratio < 0.98:
                    return 'ITM'  # In-the-money
                elif moneyness_ratio > 1.02:
                    return 'OTM'  # Out-of-the-money
                else:
                    return 'ATM'  # At-the-money
                    
        except Exception as e:
            self.logger.error(f"Error classifying option: {str(e)}")
            return 'UNKNOWN'
    
    def calculate_break_even(self, strike: float, premium: float, option_type: str) -> float:
        """Calculate break-even point for an option"""
        try:
            if option_type.upper() == 'CALL' or option_type.upper() == 'CE':
                return strike + premium
            else:  # PUT or PE
                return strike - premium
                
        except Exception as e:
            self.logger.error(f"Error calculating break-even: {str(e)}")
            return 0
    
    def get_sensitivity_analysis(self, spot: float, strike: float, expiry_date: str,
                               option_type: str, volatility: float = 0.2) -> Dict[str, Dict]:
        """Perform sensitivity analysis for different market scenarios"""
        try:
            base_greeks = self.calculate_all_greeks(spot, strike, expiry_date, option_type, volatility=volatility)
            
            scenarios = {
                'spot_up_5pct': self.calculate_all_greeks(spot * 1.05, strike, expiry_date, option_type, volatility=volatility),
                'spot_down_5pct': self.calculate_all_greeks(spot * 0.95, strike, expiry_date, option_type, volatility=volatility),
                'vol_up_5pct': self.calculate_all_greeks(spot, strike, expiry_date, option_type, volatility=volatility * 1.05),
                'vol_down_5pct': self.calculate_all_greeks(spot, strike, expiry_date, option_type, volatility=volatility * 0.95),
                'time_decay_1day': self.calculate_all_greeks(spot, strike, 
                                                           (datetime.strptime(expiry_date, '%Y-%m-%d') - pd.Timedelta(days=1)).strftime('%Y-%m-%d'), 
                                                           option_type, volatility=volatility)
            }
            
            return {
                'base_case': base_greeks,
                'scenarios': scenarios
            }
            
        except Exception as e:
            self.logger.error(f"Error in sensitivity analysis: {str(e)}")
            return {}
