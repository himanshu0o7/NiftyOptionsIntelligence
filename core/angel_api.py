import requests
import json
import pyotp
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import time
from utils.logger import Logger

class AngelOneAPI:
    """Angel One SmartAPI client for trading operations"""
    
    def __init__(self, api_key: str, client_code: str, password: str, totp: str = None):
        self.api_key = api_key
        self.client_code = client_code
        self.password = password
        self.totp = totp
        
        self.base_url = "https://apiconnect.angelone.in"
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-UserType': 'USER',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': '127.0.0.1',
            'X-ClientPublicIP': '127.0.0.1',
            'X-MACAddress': '00:00:00:00:00:00',
            'X-PrivateKey': self.api_key
        }
        
        self.jwt_token = None
        self.refresh_token = None
        self.feed_token = None
        self.session_valid = False
        
        # Required attributes for Options Greeks API
        self.client_local_ip = '127.0.0.1'
        self.client_public_ip = '127.0.0.1'
        self.mac_address = '00:00:00:00:00:00'
        
        self.logger = Logger()
    
    def connect(self) -> bool:
        """Establish connection with Angel One API"""
        try:
            # Use provided TOTP
            totp_code = self.totp
            
            login_data = {
                "clientcode": self.client_code,
                "password": self.password,
                "totp": totp_code
            }
            
            response = requests.post(
                f"{self.base_url}/rest/auth/angelbroking/user/v1/loginByPassword",
                headers=self.headers,
                data=json.dumps(login_data)
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    self.jwt_token = data['data']['jwtToken']
                    self.refresh_token = data['data']['refreshToken']
                    self.feed_token = data['data']['feedToken']
                    self.session_valid = True
                    
                    # Update headers with authorization
                    self.headers['Authorization'] = f"Bearer {self.jwt_token}"
                    
                    self.logger.info("Successfully connected to Angel One API")
                    return True
                else:
                    self.logger.error(f"Login failed: {data.get('message', 'Unknown error')}")
                    return False
            else:
                self.logger.error(f"API request failed with status code: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            return False
    
    def refresh_session(self) -> bool:
        """Refresh JWT token using refresh token"""
        try:
            refresh_data = {
                "refreshToken": self.refresh_token
            }
            
            response = requests.post(
                f"{self.base_url}/rest/auth/angelbroking/jwt/v1/generateTokens",
                headers=self.headers,
                data=json.dumps(refresh_data)
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    self.jwt_token = data['data']['jwtToken']
                    self.refresh_token = data['data']['refreshToken']
                    self.headers['Authorization'] = f"Bearer {self.jwt_token}"
                    
                    self.logger.info("Session refreshed successfully")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Session refresh error: {str(e)}")
            return False
    
    def get_profile(self) -> Optional[Dict]:
        """Get user profile information"""
        try:
            response = requests.get(
                f"{self.base_url}/rest/secure/angelbroking/user/v1/getProfile",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    return data['data']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching profile: {str(e)}")
            return None
    
    def get_ltp_data(self, exchange: str, tradingsymbol: str, symboltoken: str) -> Optional[Dict]:
        """Get Last Traded Price data"""
        try:
            ltp_data = {
                "exchange": exchange,
                "tradingsymbol": tradingsymbol,
                "symboltoken": symboltoken
            }
            
            response = requests.post(
                f"{self.base_url}/rest/secure/angelbroking/order/v1/getLTPData",
                headers=self.headers,
                data=json.dumps(ltp_data)
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    return data['data']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching LTP data: {str(e)}")
            return None
    
    def place_order(self, order_params: Dict) -> Optional[str]:
        """Place trading order"""
        try:
            response = requests.post(
                f"{self.base_url}/rest/secure/angelbroking/order/v1/placeOrder",
                headers=self.headers,
                data=json.dumps(order_params)
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    order_id = data['data']['orderid']
                    self.logger.info(f"Order placed successfully: {order_id}")
                    return order_id
                else:
                    self.logger.error(f"Order placement failed: {data.get('message')}")
                    return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return None
    
    def modify_order(self, order_params: Dict) -> bool:
        """Modify existing order"""
        try:
            response = requests.post(
                f"{self.base_url}/rest/secure/angelbroking/order/v1/modifyOrder",
                headers=self.headers,
                data=json.dumps(order_params)
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    self.logger.info(f"Order modified successfully")
                    return True
                else:
                    self.logger.error(f"Order modification failed: {data.get('message')}")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error modifying order: {str(e)}")
            return False
    
    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> bool:
        """Cancel existing order"""
        try:
            cancel_data = {
                "variety": variety,
                "orderid": order_id
            }
            
            response = requests.post(
                f"{self.base_url}/rest/secure/angelbroking/order/v1/cancelOrder",
                headers=self.headers,
                data=json.dumps(cancel_data)
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    self.logger.info(f"Order cancelled successfully: {order_id}")
                    return True
                else:
                    self.logger.error(f"Order cancellation failed: {data.get('message')}")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            return False
    
    def get_order_book(self) -> Optional[List[Dict]]:
        """Get order book"""
        try:
            response = requests.get(
                f"{self.base_url}/rest/secure/angelbroking/order/v1/getOrderBook",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    return data['data']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching order book: {str(e)}")
            return None
    
    def get_position_book(self) -> Optional[List[Dict]]:
        """Get position book"""
        try:
            response = requests.get(
                f"{self.base_url}/rest/secure/angelbroking/order/v1/getPosition",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    return data['data']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching position book: {str(e)}")
            return None
    
    def get_holdings(self) -> Optional[List[Dict]]:
        """Get holdings"""
        try:
            response = requests.get(
                f"{self.base_url}/rest/secure/angelbroking/portfolio/v1/getHolding",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    return data['data']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching holdings: {str(e)}")
            return None
    
    def get_historical_data(self, exchange: str, symboltoken: str, interval: str, 
                           from_date: str, to_date: str) -> Optional[List[Dict]]:
        """Get historical candlestick data"""
        try:
            historical_data = {
                "exchange": exchange,
                "symboltoken": symboltoken,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date
            }
            
            response = requests.post(
                f"{self.base_url}/rest/secure/angelbroking/historical/v1/getCandleData",
                headers=self.headers,
                data=json.dumps(historical_data)
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    return data['data']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            return None
    
    def get_option_chain(self, symbol: str, expiry: str) -> Optional[List[Dict]]:
        """Get options chain data (simulated using instrument master)"""
        try:
            # In a real implementation, you would fetch the instrument master
            # and filter options for the given symbol and expiry
            
            # For now, return a placeholder structure
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching option chain: {str(e)}")
            return None
    
    def logout(self) -> bool:
        """Logout from Angel One API"""
        try:
            logout_data = {
                "clientcode": self.client_code
            }
            
            response = requests.post(
                f"{self.base_url}/rest/secure/angelbroking/user/v1/logout",
                headers=self.headers,
                data=json.dumps(logout_data)
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    self.session_valid = False
                    self.jwt_token = None
                    self.refresh_token = None
                    self.feed_token = None
                    
                    self.logger.info("Logged out successfully")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error during logout: {str(e)}")
            return False
    
    def is_session_valid(self) -> bool:
        """Check if current session is valid"""
        return self.session_valid and self.jwt_token is not None
