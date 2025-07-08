import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import pytz
import json
import hashlib
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal, ROUND_HALF_UP
import requests
from utils.logger import Logger

class HelperFunctions:
    """Utility functions for the trading system"""
    
    def __init__(self):
        self.logger = Logger()
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            now = datetime.now(self.ist_timezone)
            current_time = now.time()
            current_weekday = now.weekday()
            
            # Market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
            market_start = time(9, 15)
            market_end = time(15, 30)
            
            # Check if it's a weekday (Monday=0, Friday=4)
            if current_weekday > 4:  # Saturday=5, Sunday=6
                return False
            
            # Check if current time is within market hours
            return market_start <= current_time <= market_end
            
        except Exception as e:
            self.logger.error(f"Error checking market hours: {str(e)}")
            return False
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get detailed market status"""
        try:
            now = datetime.now(self.ist_timezone)
            is_open = self.is_market_open()
            
            # Calculate next market open/close
            if is_open:
                # Market is open, calculate close time
                next_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
                if now.time() > time(15, 30):
                    next_close += timedelta(days=1)
                next_event = "Market Close"
                next_time = next_close
            else:
                # Market is closed, calculate next open
                next_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                if now.time() > time(15, 30) or now.weekday() > 4:
                    # Move to next trading day
                    days_ahead = 1
                    if now.weekday() == 4:  # Friday
                        days_ahead = 3  # Skip to Monday
                    elif now.weekday() == 5:  # Saturday
                        days_ahead = 2  # Skip to Monday
                    next_open += timedelta(days=days_ahead)
                next_event = "Market Open"
                next_time = next_open
            
            return {
                'is_open': is_open,
                'current_time': now,
                'next_event': next_event,
                'next_time': next_time,
                'time_to_next_event': next_time - now
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market status: {str(e)}")
            return {'is_open': False, 'error': str(e)}
    
    def parse_option_symbol(self, symbol: str) -> Dict[str, Any]:
        """Parse option symbol to extract underlying, expiry, strike, and type"""
        try:
            # Example: NIFTY25JAN24C21500, BANKNIFTY25JAN24P45000
            
            # Extract underlying
            if symbol.startswith('BANKNIFTY'):
                underlying = 'BANKNIFTY'
                remaining = symbol[9:]  # Remove 'BANKNIFTY'
            elif symbol.startswith('NIFTY'):
                underlying = 'NIFTY'
                remaining = symbol[5:]  # Remove 'NIFTY'
            else:
                underlying = 'UNKNOWN'
                remaining = symbol
            
            # Extract option type and strike (from the end)
            if 'CE' in remaining:
                option_type = 'CE'
                parts = remaining.split('CE')
                strike_str = parts[-1] if len(parts) > 1 else ''
            elif 'PE' in remaining:
                option_type = 'PE'
                parts = remaining.split('PE')
                strike_str = parts[-1] if len(parts) > 1 else ''
            elif remaining.endswith(('C', 'P')):
                option_type = 'CE' if remaining[-1] == 'C' else 'PE'
                # Extract numbers before the last character
                strike_match = re.search(r'(\d+)[CP]$', remaining)
                strike_str = strike_match.group(1) if strike_match else ''
            else:
                option_type = 'UNKNOWN'
                strike_str = ''
            
            # Parse strike price
            try:
                strike_price = float(strike_str) if strike_str else 0
            except ValueError:
                strike_price = 0
            
            # Extract expiry (this is simplified - in practice you'd need proper date parsing)
            expiry_pattern = r'(\d{2}[A-Z]{3}\d{2})'
            expiry_match = re.search(expiry_pattern, symbol)
            expiry_str = expiry_match.group(1) if expiry_match else ''
            
            return {
                'underlying': underlying,
                'expiry': expiry_str,
                'strike_price': strike_price,
                'option_type': option_type,
                'original_symbol': symbol,
                'is_valid': underlying != 'UNKNOWN' and option_type != 'UNKNOWN' and strike_price > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing option symbol {symbol}: {str(e)}")
            return {
                'underlying': 'UNKNOWN',
                'expiry': '',
                'strike_price': 0,
                'option_type': 'UNKNOWN',
                'original_symbol': symbol,
                'is_valid': False
            }
    
    def calculate_lot_size(self, symbol: str) -> int:
        """Get lot size for a symbol"""
        try:
            symbol_upper = symbol.upper()
            
            if 'NIFTY' in symbol_upper and 'BANKNIFTY' not in symbol_upper:
                return 50  # NIFTY lot size
            elif 'BANKNIFTY' in symbol_upper:
                return 15  # BANKNIFTY lot size
            elif 'FINNIFTY' in symbol_upper:
                return 40  # FINNIFTY lot size
            elif 'MIDCPNIFTY' in symbol_upper:
                return 75  # MIDCPNIFTY lot size
            else:
                return 1  # Default for equity
                
        except Exception as e:
            self.logger.error(f"Error calculating lot size for {symbol}: {str(e)}")
            return 1
    
    def round_to_tick_size(self, price: float, symbol: str) -> float:
        """Round price to valid tick size"""
        try:
            # Define tick sizes based on price ranges
            if price < 1:
                tick_size = 0.0025
            elif price < 5:
                tick_size = 0.0050
            elif price < 20:
                tick_size = 0.0100
            elif price < 200:
                tick_size = 0.0500
            else:
                tick_size = 0.0500
            
            # Round to nearest tick
            return float(Decimal(str(price / tick_size)).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * tick_size)
            
        except Exception as e:
            self.logger.error(f"Error rounding price {price} for {symbol}: {str(e)}")
            return price
    
    def validate_order_params(self, order_params: Dict) -> Tuple[bool, str]:
        """Validate order parameters"""
        try:
            required_fields = ['symbol', 'quantity', 'price', 'transaction_type', 'order_type', 'product_type']
            
            # Check required fields
            for field in required_fields:
                if field not in order_params:
                    return False, f"Missing required field: {field}"
                
                if order_params[field] is None or order_params[field] == '':
                    return False, f"Empty value for required field: {field}"
            
            # Validate transaction type
            if order_params['transaction_type'].upper() not in ['BUY', 'SELL']:
                return False, "Invalid transaction_type. Must be 'BUY' or 'SELL'"
            
            # Validate order type
            valid_order_types = ['MARKET', 'LIMIT', 'SL', 'SL-M']
            if order_params['order_type'].upper() not in valid_order_types:
                return False, f"Invalid order_type. Must be one of {valid_order_types}"
            
            # Validate product type
            valid_product_types = ['DELIVERY', 'INTRADAY', 'CARRYFORWARD', 'BO', 'CO']
            if order_params['product_type'].upper() not in valid_product_types:
                return False, f"Invalid product_type. Must be one of {valid_product_types}"
            
            # Validate numeric fields
            try:
                quantity = int(order_params['quantity'])
                if quantity <= 0:
                    return False, "Quantity must be positive"
                    
                price = float(order_params['price'])
                if price <= 0:
                    return False, "Price must be positive"
                    
            except (ValueError, TypeError):
                return False, "Quantity must be integer, price must be numeric"
            
            return True, "Valid order parameters"
            
        except Exception as e:
            self.logger.error(f"Error validating order params: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def calculate_margin_required(self, symbol: str, quantity: int, price: float, 
                                 transaction_type: str, product_type: str) -> float:
        """Calculate margin required for a trade (simplified)"""
        try:
            # This is a simplified margin calculation
            # In practice, you'd use Angel One's margin calculator API
            
            position_value = quantity * price
            
            if product_type.upper() == 'DELIVERY':
                # Delivery requires full payment for buy orders
                if transaction_type.upper() == 'BUY':
                    return position_value
                else:
                    return 0  # Short selling in delivery not typical
            
            elif product_type.upper() == 'INTRADAY':
                # Intraday margin is typically 10-20% of position value
                margin_percentage = 0.15  # 15%
                return position_value * margin_percentage
            
            elif product_type.upper() == 'CARRYFORWARD':
                # F&O margin varies by underlying
                if 'NIFTY' in symbol.upper():
                    margin_percentage = 0.12  # 12% for index options
                else:
                    margin_percentage = 0.20  # 20% for others
                return position_value * margin_percentage
            
            else:
                # Default margin
                return position_value * 0.15
                
        except Exception as e:
            self.logger.error(f"Error calculating margin: {str(e)}")
            return 0.0
    
    def format_currency(self, amount: float, currency: str = '₹') -> str:
        """Format amount as currency"""
        try:
            if abs(amount) >= 10000000:  # 1 crore
                return f"{currency}{amount/10000000:.2f}Cr"
            elif abs(amount) >= 100000:  # 1 lakh
                return f"{currency}{amount/100000:.2f}L"
            elif abs(amount) >= 1000:  # 1 thousand
                return f"{currency}{amount/1000:.2f}K"
            else:
                return f"{currency}{amount:,.2f}"
                
        except Exception as e:
            self.logger.error(f"Error formatting currency: {str(e)}")
            return f"{currency}{amount}"
    
    def calculate_percentage_change(self, old_value: float, new_value: float) -> float:
        """Calculate percentage change"""
        try:
            if old_value == 0:
                return 0.0
            return ((new_value - old_value) / old_value) * 100
        except Exception as e:
            self.logger.error(f"Error calculating percentage change: {str(e)}")
            return 0.0
    
    def generate_signal_id(self, strategy_name: str, symbol: str, timestamp: datetime = None) -> str:
        """Generate unique signal ID"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Create unique string
            unique_string = f"{strategy_name}_{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Generate hash
            signal_id = hashlib.md5(unique_string.encode()).hexdigest()[:12]
            
            return f"{strategy_name[:3].upper()}_{signal_id}"
            
        except Exception as e:
            self.logger.error(f"Error generating signal ID: {str(e)}")
            return f"SIG_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value"""
        try:
            if denominator == 0:
                return default
            return numerator / denominator
        except Exception as e:
            self.logger.error(f"Error in safe divide: {str(e)}")
            return default
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file operations"""
        try:
            # Remove invalid characters
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            
            # Remove leading/trailing spaces and dots
            filename = filename.strip(' .')
            
            # Limit length
            if len(filename) > 255:
                filename = filename[:255]
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Error sanitizing filename: {str(e)}")
            return "unknown_file"
    
    def convert_timeframe(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        try:
            timeframe_map = {
                '1m': 1,
                '3m': 3,
                '5m': 5,
                '15m': 15,
                '30m': 30,
                '1h': 60,
                '2h': 120,
                '4h': 240,
                '1d': 1440
            }
            
            return timeframe_map.get(timeframe.lower(), 5)  # Default to 5 minutes
            
        except Exception as e:
            self.logger.error(f"Error converting timeframe: {str(e)}")
            return 5
    
    def is_business_day(self, date: datetime) -> bool:
        """Check if date is a business day (Monday-Friday)"""
        try:
            return date.weekday() < 5
        except Exception as e:
            self.logger.error(f"Error checking business day: {str(e)}")
            return False
    
    def get_next_business_day(self, date: datetime) -> datetime:
        """Get next business day"""
        try:
            next_day = date + timedelta(days=1)
            while not self.is_business_day(next_day):
                next_day += timedelta(days=1)
            return next_day
        except Exception as e:
            self.logger.error(f"Error getting next business day: {str(e)}")
            return date + timedelta(days=1)
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol format is correct"""
        try:
            if not symbol or len(symbol) < 3:
                return False
            
            # Check for valid characters (alphanumeric and specific symbols)
            if not re.match(r'^[A-Z0-9\-&]+$', symbol.upper()):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating symbol: {str(e)}")
            return False
    
    def get_expiry_dates(self, underlying: str, current_date: datetime = None) -> List[str]:
        """Get list of expiry dates for an underlying"""
        try:
            if current_date is None:
                current_date = datetime.now()
            
            expiry_dates = []
            
            # For index options, expiry is typically on Thursdays
            # Find next 4-5 Thursdays
            days_ahead = (3 - current_date.weekday()) % 7  # Next Thursday
            if days_ahead == 0 and current_date.hour >= 15:  # After market close on Thursday
                days_ahead = 7
            
            for i in range(5):  # Next 5 expiries
                expiry_date = current_date + timedelta(days=days_ahead + (i * 7))
                expiry_dates.append(expiry_date.strftime('%Y-%m-%d'))
            
            return expiry_dates
            
        except Exception as e:
            self.logger.error(f"Error getting expiry dates: {str(e)}")
            return []
    
    def chunks(self, lst: List, chunk_size: int) -> List[List]:
        """Split list into chunks of specified size"""
        try:
            return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
        except Exception as e:
            self.logger.error(f"Error creating chunks: {str(e)}")
            return [lst]
    
    def retry_on_failure(self, func, max_retries: int = 3, delay: float = 1.0):
        """Retry function on failure"""
        import time
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Function failed after {max_retries} attempts: {str(e)}")
                    raise
                else:
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}")
                    time.sleep(delay)
    
    def deep_merge_dicts(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries"""
        try:
            result = dict1.copy()
            
            for key, value in dict2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self.deep_merge_dicts(result[key], value)
                else:
                    result[key] = value
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error merging dictionaries: {str(e)}")
            return dict1

# Create helper instance
helper = HelperFunctions()
