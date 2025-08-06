import pandas as pd
from datetime import datetime, date
from typing import Union, Any

class Helper:
    """Helper utilities for the trading system"""

    @staticmethod
    def format_currency(amount: Union[float, int]) -> str:
        """Format amount as Indian currency"""
        if amount is None:
            return "₹0"

        try:
            if amount >= 0:
                return f"₹{amount:,.2f}"
            else:
                return f"-₹{abs(amount):,.2f}"
        except:
            return "₹0"

    @staticmethod
    def format_percentage(value: Union[float, int]) -> str:
        """Format value as percentage"""
        if value is None:
            return "0%"

        try:
            return f"{value:.2f}%"
        except:
            return "0%"

    @staticmethod
    def is_business_day(date_obj: Union[datetime, date]) -> bool:
        """Check if date is a business day (Mon-Fri)"""
        try:
            if isinstance(date_obj, datetime):
                return date_obj.weekday() < 5
            elif isinstance(date_obj, date):
                return date_obj.weekday() < 5
            else:
                return True
        except:
            return True

    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert value to float"""
        try:
            return float(value)
        except:
            return default

    @staticmethod
    def safe_int(value: Any, default: int = 0) -> int:
        """Safely convert value to int"""
        try:
            return int(value)
        except:
            return default

    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """Calculate percentage change between two values"""
        try:
            if old_value == 0:
                return 0.0
            return ((new_value - old_value) / old_value) * 100
        except:
            return 0.0

    @staticmethod
    def format_timestamp(timestamp: Union[datetime, str]) -> str:
        """Format timestamp for display"""
        try:
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            return timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return str(timestamp)

# Create a global helper instance
helper = Helper()