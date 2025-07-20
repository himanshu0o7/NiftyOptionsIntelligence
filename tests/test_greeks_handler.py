"""
Comprehensive test suite for greeks_handler.py

This module contains both unit tests and integration tests for the Greeks handler functionality,
including tests for edge cases, error conditions, and positive flows.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import json
import threading
import time
import pandas as pd
from io import StringIO
import sys
import os
import urllib.request

# Add the parent directory to the path to import greeks_handler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import greeks_handler


class TestGreeksHandlerUnit(unittest.TestCase):
    """Unit tests for individual functions in greeks_handler.py"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Clear global variables before each test
        greeks_handler.option_data = {}
        greeks_handler.symbol_dict = {}
        greeks_handler.sws = None
    
    def tearDown(self):
        """Clean up after each test method."""
        # Reset global variables
        greeks_handler.option_data = {}
        greeks_handler.symbol_dict = {}
        greeks_handler.sws = None


class TestGetOptionGreeksDummy(TestGreeksHandlerUnit):
    """Test cases for the dummy get_option_greeks function"""
    
    def test_valid_call_option_greeks(self):
        """Test dummy function with valid CE option parameters"""
        result = greeks_handler.get_option_greeks("NIFTY", 18000, "CE")
        
        self.assertIsInstance(result, dict)
        self.assertNotIn("error", result)
        self.assertEqual(result["symbol"], "NIFTY25JUL18000CE")
        self.assertEqual(result["strike"], 18000.0)
        self.assertEqual(result["option_type"], "CE")
        self.assertEqual(result["delta"], 0.5)
        self.assertIsInstance(result["ltp"], float)
        self.assertIsInstance(result["iv"], float)
        self.assertIsInstance(result["gamma"], float)
        self.assertIsInstance(result["theta"], float)
        self.assertIsInstance(result["vega"], float)
    
    def test_valid_put_option_greeks(self):
        """Test dummy function with valid PE option parameters"""
        result = greeks_handler.get_option_greeks("BANKNIFTY", 45000, "PE")
        
        self.assertIsInstance(result, dict)
        self.assertNotIn("error", result)
        self.assertEqual(result["symbol"], "BANKNIFTY25JUL45000PE")
        self.assertEqual(result["strike"], 45000.0)
        self.assertEqual(result["option_type"], "PE")
        self.assertEqual(result["delta"], -0.5)
    
    def test_invalid_symbol_none(self):
        """Test dummy function with None symbol"""
        result = greeks_handler.get_option_greeks(None, 18000, "CE")
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid symbol provided")
    
    def test_invalid_symbol_empty_string(self):
        """Test dummy function with empty string symbol"""
        result = greeks_handler.get_option_greeks("", 18000, "CE")
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid symbol provided")
    
    def test_invalid_symbol_non_string(self):
        """Test dummy function with non-string symbol"""
        result = greeks_handler.get_option_greeks(123, 18000, "CE")
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid symbol provided")
    
    def test_invalid_strike_negative(self):
        """Test dummy function with negative strike price"""
        result = greeks_handler.get_option_greeks("NIFTY", -18000, "CE")
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid strike price provided")
    
    def test_invalid_strike_zero(self):
        """Test dummy function with zero strike price"""
        result = greeks_handler.get_option_greeks("NIFTY", 0, "CE")
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid strike price provided")
    
    def test_invalid_strike_non_numeric(self):
        """Test dummy function with non-numeric strike price"""
        result = greeks_handler.get_option_greeks("NIFTY", "invalid", "CE")
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid strike price provided")
    
    def test_invalid_option_type_call(self):
        """Test dummy function with invalid option type"""
        result = greeks_handler.get_option_greeks("NIFTY", 18000, "CALL")
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid option type. Must be 'CE' or 'PE'")
    
    def test_invalid_option_type_put(self):
        """Test dummy function with invalid option type"""
        result = greeks_handler.get_option_greeks("NIFTY", 18000, "PUT")
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid option type. Must be 'CE' or 'PE'")
    
    def test_invalid_option_type_none(self):
        """Test dummy function with None option type"""
        result = greeks_handler.get_option_greeks("NIFTY", 18000, None)
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid option type. Must be 'CE' or 'PE'")


class TestFetchOptionGreeksUnit(TestGreeksHandlerUnit):
    """Unit tests for fetch_option_greeks function with mocked dependencies"""
    
    def setUp(self):
        """Set up mocks for each test"""
        super().setUp()
        
        # Sample instrument data
        self.sample_instruments = [
            {
                "name": "NIFTY",
                "instrumenttype": "OPTIDX",
                "token": "12345",
                "symbol": "NIFTY25JUL18000CE",
                "strike": "1800000",  # Strike in paise
                "expiry": "25JUL2025"
            },
            {
                "name": "NIFTY", 
                "instrumenttype": "OPTIDX",
                "token": "12346",
                "symbol": "NIFTY25JUL18000PE",
                "strike": "1800000",  # Strike in paise
                "expiry": "25JUL2025"
            }
        ]
        
        # Sample Greeks data
        self.sample_greeks = [
            {
                "strikePrice": "18000",
                "optionType": "CE",
                "impliedVolatility": "15.25",
                "delta": "0.5",
                "gamma": "0.02",
                "theta": "-2.5",
                "vega": "12.8"
            }
        ]
        
        # Sample tokens
        self.sample_tokens = {
            "api_key": "test_api_key",
            "jwtToken": "test_jwt_token",
            "feedToken": "test_feed_token",
            "clientcode": "test_client_code"
        }
    
    @patch('greeks_handler.SmartConnect')
    @patch('greeks_handler.SmartWebSocketV2')
    @patch('urllib.request.urlopen')
    @patch('greeks_handler.threading.Thread')
    @patch('greeks_handler.time.sleep')
    def test_fetch_option_greeks_success(self, mock_sleep, mock_thread, mock_urlopen, mock_ws, mock_smart_connect):
        """Test successful fetch of option greeks"""
        
        # Mock urllib response
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(self.sample_instruments).encode()
        mock_urlopen.return_value = mock_response
        
        # Mock SmartConnect
        mock_obj = Mock()
        mock_obj.optionGreek.return_value = {"data": self.sample_greeks}
        mock_smart_connect.return_value = mock_obj
        
        # Mock WebSocket
        mock_ws_instance = Mock()
        mock_ws.return_value = mock_ws_instance
        
        # Mock thread
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        # Set up option_data with mock live data
        greeks_handler.option_data["NIFTY25JUL18000CE"] = {
            "ltp": 100.50,
            "oi": 50000,
            "volume": 10000,
            "bid": 99.50,
            "ask": 101.50
        }
        
        result = greeks_handler.fetch_option_greeks("NIFTY", 18000, "CE", self.sample_tokens)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertNotIn("error", result)
        self.assertEqual(result["symbol"], "NIFTY25JUL18000CE")
        self.assertEqual(result["strike"], 18000)
        self.assertEqual(result["option_type"], "CE")
        self.assertEqual(result["expiry"], "25JUL2025")
        
        # Verify API calls
        mock_smart_connect.assert_called_once_with(api_key="test_api_key")
        mock_obj.optionGreek.assert_called_once()
        mock_ws.assert_called_once()
        mock_thread.assert_called_once()
    
    @patch('urllib.request.urlopen')
    def test_fetch_option_greeks_symbol_not_found(self, mock_urlopen):
        """Test fetch with symbol not found in instrument list"""
        
        # Mock urllib response with empty instruments
        mock_response = Mock()
        mock_response.read.return_value = json.dumps([]).encode()
        mock_urlopen.return_value = mock_response
        
        result = greeks_handler.fetch_option_greeks("INVALID", 18000, "CE", self.sample_tokens)
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Symbol not found in instrument list")
    
    @patch('urllib.request.urlopen')
    def test_fetch_option_greeks_contract_not_found(self, mock_urlopen):
        """Test fetch with no matching contract for strike/option_type"""
        
        # Mock urllib response with instruments but different strikes
        instruments_different_strike = [
            {
                "name": "NIFTY",
                "instrumenttype": "OPTIDX",
                "token": "12345",
                "symbol": "NIFTY25JUL19000CE",
                "strike": "1900000",  # Different strike
                "expiry": "25JUL2025"
            }
        ]
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(instruments_different_strike).encode()
        mock_urlopen.return_value = mock_response
        
        result = greeks_handler.fetch_option_greeks("NIFTY", 18000, "CE", self.sample_tokens)
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "No CE contract found for NIFTY 18000")
    
    @patch('greeks_handler.SmartConnect')
    @patch('greeks_handler.SmartWebSocketV2')
    @patch('urllib.request.urlopen')
    @patch('greeks_handler.threading.Thread')
    @patch('greeks_handler.time.sleep')
    def test_fetch_option_greeks_no_greeks_data(self, mock_sleep, mock_thread, mock_urlopen, mock_ws, mock_smart_connect):
        """Test fetch with no matching Greeks data"""
        
        # Mock urllib response
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(self.sample_instruments).encode()
        mock_urlopen.return_value = mock_response
        
        # Mock SmartConnect with empty Greeks data
        mock_obj = Mock()
        mock_obj.optionGreek.return_value = {"data": []}
        mock_smart_connect.return_value = mock_obj
        
        # Mock WebSocket
        mock_ws_instance = Mock()
        mock_ws.return_value = mock_ws_instance
        
        # Mock thread
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        result = greeks_handler.fetch_option_greeks("NIFTY", 18000, "CE", self.sample_tokens)
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "No matching Greek data")
    
    @patch('urllib.request.urlopen')
    def test_fetch_option_greeks_network_error(self, mock_urlopen):
        """Test fetch with network error when fetching instruments"""
        
        mock_urlopen.side_effect = Exception("Network error")
        
        with self.assertRaises(Exception):
            greeks_handler.fetch_option_greeks("NIFTY", 18000, "CE", self.sample_tokens)
    
    @patch('greeks_handler.SmartConnect')
    @patch('urllib.request.urlopen')
    def test_fetch_option_greeks_smart_connect_error(self, mock_urlopen, mock_smart_connect):
        """Test fetch with SmartConnect API error"""
        
        # Mock urllib response
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(self.sample_instruments).encode()
        mock_urlopen.return_value = mock_response
        
        # Mock SmartConnect to raise an exception
        mock_smart_connect.side_effect = Exception("API connection failed")
        
        with self.assertRaises(Exception):
            greeks_handler.fetch_option_greeks("NIFTY", 18000, "CE", self.sample_tokens)


class TestGreeksHandlerIntegration(unittest.TestCase):
    """Integration tests for greeks_handler.py functions working together"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Clear global variables
        greeks_handler.option_data = {}
        greeks_handler.symbol_dict = {}
        greeks_handler.sws = None
    
    def tearDown(self):
        """Clean up after tests"""
        # Reset global variables
        greeks_handler.option_data = {}
        greeks_handler.symbol_dict = {}
        greeks_handler.sws = None
    
    def test_global_variable_persistence(self):
        """Test that global variables maintain state"""
        # Test option_data persistence
        test_data = {"test_symbol": {"ltp": 100}}
        greeks_handler.option_data.update(test_data)
        
        self.assertEqual(greeks_handler.option_data["test_symbol"]["ltp"], 100)
        
        # Test symbol_dict persistence
        test_symbols = {"NIFTY": "12345"}
        greeks_handler.symbol_dict.update(test_symbols)
        
        self.assertEqual(greeks_handler.symbol_dict["NIFTY"], "12345")
    
    @patch('greeks_handler.SmartConnect')
    @patch('greeks_handler.SmartWebSocketV2')
    @patch('urllib.request.urlopen')
    @patch('greeks_handler.threading.Thread')
    @patch('greeks_handler.time.sleep')
    def test_websocket_data_flow(self, mock_sleep, mock_thread, mock_urlopen, mock_ws, mock_smart_connect):
        """Test the flow of data from WebSocket to final result"""
        
        # Setup instrument data
        instruments = [
            {
                "name": "NIFTY",
                "instrumenttype": "OPTIDX",
                "token": "12345",
                "symbol": "NIFTY25JUL18000CE",
                "strike": "1800000",
                "expiry": "25JUL2025"
            }
        ]
        
        # Mock urllib response
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(instruments).encode()
        mock_urlopen.return_value = mock_response
        
        # Mock SmartConnect
        mock_obj = Mock()
        greeks_data = [
            {
                "strikePrice": "18000",
                "optionType": "CE", 
                "impliedVolatility": "15.25",
                "delta": "0.5",
                "gamma": "0.02",
                "theta": "-2.5",
                "vega": "12.8"
            }
        ]
        mock_obj.optionGreek.return_value = {"data": greeks_data}
        mock_smart_connect.return_value = mock_obj
        
        # Mock WebSocket with data callback simulation
        mock_ws_instance = Mock()
        mock_ws.return_value = mock_ws_instance
        
        def simulate_websocket_callback():
            """Simulate WebSocket data reception"""
            # Simulate the on_data callback
            message = {
                "last_traded_price": 100.50,
                "open_interest": 50000,
                "volume_trade_for_the_day": 10000,
                "best_5_buy_data": [{"price": 99.50}],
                "best_5_sell_data": [{"price": 101.50}]
            }
            # Manually populate option_data as the callback would
            greeks_handler.option_data["NIFTY25JUL18000CE"] = {
                "ltp": message["last_traded_price"],
                "oi": message["open_interest"], 
                "volume": message["volume_trade_for_the_day"],
                "bid": message["best_5_buy_data"][0]["price"],
                "ask": message["best_5_sell_data"][0]["price"]
            }
        
        # Execute the simulation
        simulate_websocket_callback()
        
        tokens = {
            "api_key": "test_api_key",
            "jwtToken": "test_jwt_token", 
            "feedToken": "test_feed_token",
            "clientcode": "test_client_code"
        }
        
        result = greeks_handler.fetch_option_greeks("NIFTY", 18000, "CE", tokens)
        
        # Verify the integrated result contains both WebSocket and API data
        self.assertIsInstance(result, dict)
        self.assertNotIn("error", result)
        self.assertEqual(result["ltp"], 100.50)  # From WebSocket
        self.assertEqual(result["iv"], "15.25")  # From Greeks API
        self.assertEqual(result["delta"], "0.5")  # From Greeks API
        self.assertEqual(result["oi"], 50000)  # From WebSocket
    
    def test_multiple_symbol_processing(self):
        """Test processing multiple symbols with dummy function"""
        symbols = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
        strikes = [18000, 45000, 19000]
        option_types = ["CE", "PE", "CE"]
        
        results = []
        for symbol, strike, option_type in zip(symbols, strikes, option_types):
            result = greeks_handler.get_option_greeks(symbol, strike, option_type)
            results.append(result)
        
        # Verify all results are valid
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertNotIn("error", result)
            self.assertIn("symbol", result)
            self.assertIn("strike", result)
            self.assertIn("option_type", result)
    
    def test_error_propagation(self):
        """Test that errors are properly propagated through the system"""
        # Test with invalid inputs to dummy function
        invalid_cases = [
            (None, 18000, "CE"),
            ("NIFTY", -18000, "CE"),
            ("NIFTY", 18000, "INVALID")
        ]
        
        for symbol, strike, option_type in invalid_cases:
            result = greeks_handler.get_option_greeks(symbol, strike, option_type)
            self.assertIn("error", result)
            self.assertIsInstance(result["error"], str)


class TestGreeksHandlerEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_dummy_function_boundary_values(self):
        """Test dummy function with boundary values"""
        # Test minimum valid strike
        result = greeks_handler.get_option_greeks("NIFTY", 0.01, "CE")
        self.assertNotIn("error", result)
        self.assertEqual(result["strike"], 0.01)
        
        # Test very large strike
        result = greeks_handler.get_option_greeks("NIFTY", 999999, "CE")
        self.assertNotIn("error", result)
        self.assertEqual(result["strike"], 999999.0)
        
        # Test float strike
        result = greeks_handler.get_option_greeks("NIFTY", 18000.5, "CE")
        self.assertNotIn("error", result)
        self.assertEqual(result["strike"], 18000.5)
    
    def test_dummy_function_symbol_variations(self):
        """Test dummy function with various symbol formats"""
        valid_symbols = ["NIFTY", "BANKNIFTY", "FINNIFTY", "nifty", "MIDCPNIFTY"]
        
        for symbol in valid_symbols:
            result = greeks_handler.get_option_greeks(symbol, 18000, "CE")
            self.assertNotIn("error", result)
            self.assertIn(symbol, result["symbol"])
    
    def test_global_variable_thread_safety_simulation(self):
        """Simulate thread safety concerns with global variables"""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                # Simulate concurrent access to global variables
                greeks_handler.option_data[f"symbol_{worker_id}"] = {"ltp": worker_id * 100}
                time.sleep(0.01)  # Small delay to increase chance of race conditions
                value = greeks_handler.option_data.get(f"symbol_{worker_id}", {}).get("ltp")
                results.append((worker_id, value))
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify no errors occurred and all data was written
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 5)
        
        # Verify data integrity
        for worker_id, value in results:
            expected_value = worker_id * 100
            self.assertEqual(value, expected_value, f"Data mismatch for worker {worker_id}")


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)