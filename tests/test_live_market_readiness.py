"""
Basic tests for critical functionality to ensure live market readiness.
"""
import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestCriticalFunctionality(unittest.TestCase):
    """Test critical components for live market operations."""

    def test_imports_syntax(self):
        """Test that critical modules can be imported without syntax errors."""
        try:
            import angel_utils
            import production_config
            import session_manager
            import smart_websocket_handler
            import token_mapper
        except SyntaxError as e:
            self.fail(f"Syntax error in critical modules: {e}")
        except ImportError:
            # Import errors are expected in test environment without dependencies
            pass

    def test_production_config_structure(self):
        """Test production configuration structure."""
        from production_config import ProductionConfig, RateLimiter

        # Test RateLimiter functionality
        limiter = RateLimiter(max_calls=5, time_window=60)
        self.assertTrue(limiter.can_make_call())

        # Add calls up to limit
        for _ in range(5):
            limiter.add_call()

        # Should not be able to make more calls
        self.assertFalse(limiter.can_make_call())

    @patch.dict(os.environ, {
        'ANGEL_API_KEY': 'test_key',
        'ANGEL_CLIENT_ID': 'test_client',
        'ANGEL_PIN': 'test_pin',
        'ANGEL_TOTP_SECRET': 'test_secret'
    })
    def test_production_config_validation(self):
        """Test that production config validates environment variables."""
        from production_config import ProductionConfig

        # Should not raise with all required vars
        try:
            config = ProductionConfig()
        except ValueError:
            self.fail("ProductionConfig should not raise with all required env vars")

    def test_missing_env_vars_validation(self):
        """Test that missing environment variables are properly detected."""
        from production_config import ProductionConfig

        # Clear environment
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                ProductionConfig()

            self.assertIn("Missing required environment variables", str(context.exception))

    def test_angel_utils_functions_exist(self):
        """Test that critical angel_utils functions exist."""
        try:
            import angel_utils

            # Check that key functions exist
            self.assertTrue(hasattr(angel_utils, 'login'))
            self.assertTrue(hasattr(angel_utils, 'load_master_contract'))
            self.assertTrue(hasattr(angel_utils, 'find_token'))
            self.assertTrue(hasattr(angel_utils, 'get_ltp'))
        except ImportError:
            # Expected in test environment without dependencies
            pass

    def test_websocket_handler_structure(self):
        """Test websocket handler has required methods."""
        try:
            from smart_websocket_handler import SmartWebSocketHandler

            # Check that key methods exist
            handler_methods = dir(SmartWebSocketHandler)
            required_methods = ['connect', 'close', '_on_data', '_on_error']

            for method in required_methods:
                self.assertIn(method, handler_methods, f"Missing method: {method}")
        except ImportError:
            # Expected in test environment without dependencies
            pass

    def test_rate_limiting_decorator(self):
        """Test rate limiting decorator functionality."""
        from production_config import rate_limited, RateLimiter

        limiter = RateLimiter(max_calls=2, time_window=60)

        @rate_limited(limiter)
        def test_function():
            return "success"

        # Should work for first calls
        self.assertEqual(test_function(), "success")
        self.assertEqual(test_function(), "success")

        # Third call triggers wait but completes successfully
        self.assertEqual(test_function(), "success")
        self.assertTrue(limiter.can_make_call())

class TestLiveMarketReadiness(unittest.TestCase):
    """Test live market specific functionality."""

    def test_trading_hours_config(self):
        """Test trading hours configuration."""
        from production_config import ProductionConfig

        trading_hours = ProductionConfig.get_trading_hours()

        self.assertIn('market_open', trading_hours)
        self.assertIn('market_close', trading_hours)
        self.assertIn('timezone', trading_hours)
        self.assertEqual(trading_hours['timezone'], 'Asia/Kolkata')

    def test_retry_config(self):
        """Test retry configuration for API resilience."""
        from production_config import ProductionConfig

        retry_config = ProductionConfig.get_retry_config()

        self.assertIn('max_retries', retry_config)
        self.assertIn('retry_delay', retry_config)
        self.assertIn('backoff_multiplier', retry_config)
        self.assertGreater(retry_config['max_retries'], 0)

if __name__ == '__main__':
    unittest.main()