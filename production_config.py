# production_config.py
"""
Production configuration and rate limiting for live market operations.
"""

import os
import time
import logging
from functools import wraps
from typing import Optional, Dict, Any

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API calls to prevent exceeding limits."""

    def __init__(self, max_calls: int = 100, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []

    def can_make_call(self) -> bool:
        """Check if we can make a call without exceeding rate limits."""
        now = time.time()
        # Remove calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]

        return len(self.calls) < self.max_calls

    def add_call(self):
        """Record a new API call."""
        self.calls.append(time.time())

    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        if not self.can_make_call():
            sleep_time = self.time_window - (time.time() - self.calls[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

# Global rate limiter instances
angel_api_limiter = RateLimiter(max_calls=100, time_window=60)  # Angel One limits
yahoo_finance_limiter = RateLimiter(max_calls=2000, time_window=3600)  # Yahoo Finance limits

def rate_limited(limiter: RateLimiter):
    """Decorator to apply rate limiting to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.wait_if_needed()
            try:
                result = func(*args, **kwargs)
                limiter.add_call()
                return result
            except Exception as e:
                logger.error(f"Rate limited function {func.__name__} failed: {e}")
                raise
        return wrapper
    return decorator

class ProductionConfig:
    """Production configuration with environment variable validation."""

    def __init__(self):
        self.validate_required_env_vars()
        self.setup_logging()

    @staticmethod
    def validate_required_env_vars():
        """Validate that all required environment variables are set."""
        required_vars = [
            'ANGEL_API_KEY',
            'ANGEL_CLIENT_ID',
            'ANGEL_PIN',
            'ANGEL_TOTP_SECRET'
        ]

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("✅ All required environment variables are set")

    @staticmethod
    def setup_logging():
        """Setup production logging configuration."""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        numeric_level = getattr(logging, log_level, None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {log_level}')

        logging.getLogger().setLevel(numeric_level)
        logger.info(f"✅ Logging configured at {log_level} level")

    @staticmethod
    def get_trading_hours() -> Dict[str, str]:
        """Get trading hours configuration."""
        return {
            'market_open': '09:15',
            'market_close': '15:30',
            'timezone': 'Asia/Kolkata'
        }

    @staticmethod
    def get_retry_config() -> Dict[str, Any]:
        """Get retry configuration for API calls."""
        return {
            'max_retries': 3,
            'retry_delay': 1,  # seconds
            'backoff_multiplier': 2
        }

def with_retry(max_retries: int = 3, delay: float = 1, backoff: float = 2):
    """Decorator to add retry logic to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = delay * (backoff ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")

            raise last_exception
        return wrapper
    return decorator

# Export commonly used instances - only create when explicitly needed
def get_config():
    """Get production config instance (lazy loading)."""
    return ProductionConfig()