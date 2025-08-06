import logging
import os
from datetime import datetime
from typing import Optional
import sys

class Logger:
    """Centralized logging system for the trading application"""

    def __init__(self, name: str = 'TradingSystem', log_level: str = 'INFO',
                 log_file: str = 'trading_system.log'):
        self.name = name
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_file = log_file

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()

    def _setup_handlers(self):
        """Setup file and console handlers"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, **kwargs)

    def log_trade(self, symbol: str, action: str, quantity: int, price: float,
                  strategy: str, order_id: str = None):
        """Log trading activity"""
        trade_msg = f"TRADE - {action} {quantity} {symbol} @ {price} | Strategy: {strategy}"
        if order_id:
            trade_msg += f" | Order ID: {order_id}"
        self.info(trade_msg)

    def log_signal(self, symbol: str, signal_type: str, action: str,
                   confidence: float, strategy: str):
        """Log trading signals"""
        signal_msg = f"SIGNAL - {signal_type} {action} {symbol} | Confidence: {confidence:.2%} | Strategy: {strategy}"
        self.info(signal_msg)

    def log_pnl(self, symbol: str, pnl: float, pnl_pct: float, strategy: str):
        """Log P&L updates"""
        pnl_msg = f"P&L - {symbol}: ₹{pnl:,.2f} ({pnl_pct:+.2%}) | Strategy: {strategy}"
        self.info(pnl_msg)

    def log_risk_warning(self, warning_type: str, details: str):
        """Log risk warnings"""
        risk_msg = f"RISK WARNING - {warning_type}: {details}"
        self.warning(risk_msg)

    def log_system_event(self, event_type: str, details: str):
        """Log system events"""
        system_msg = f"SYSTEM - {event_type}: {details}"
        self.info(system_msg)

    def log_api_error(self, api_name: str, error_code: str, error_message: str):
        """Log API errors"""
        api_msg = f"API ERROR - {api_name} [{error_code}]: {error_message}"
        self.error(api_msg)

    def log_performance(self, function_name: str, execution_time: float):
        """Log performance metrics"""
        perf_msg = f"PERFORMANCE - {function_name} executed in {execution_time:.4f}s"
        self.debug(perf_msg)

    def set_level(self, level: str):
        """Change logging level"""
        new_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(new_level)
        for handler in self.logger.handlers:
            handler.setLevel(new_level)

    @staticmethod
    def get_logger(name: str = 'TradingSystem') -> 'Logger':
        """Get logger instance"""
        return Logger(name)

# Create default logger instance
default_logger = Logger()
