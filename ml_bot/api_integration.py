"""
API Integration Module
Handles communication between ML Bot and Main Trading System
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, Optional
from datetime import datetime

class APIIntegration:
    """Handle API communication with main trading system"""

    def __init__(self, websocket_url: str, http_api_url: str):
        self.websocket_url = websocket_url
        self.http_api_url = http_api_url
        self.logger = logging.getLogger(__name__)
        self.websocket_connection = None

    async def connect_websocket(self):
        """Establish WebSocket connection"""
        try:
            self.websocket_connection = await websockets.connect(self.websocket_url)
            self.logger.info("WebSocket connected successfully")
            return True
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            return False

    async def send_ml_signal(self, signal_data: Dict):
        """Send ML signal to main system"""
        try:
            if self.websocket_connection:
                await self.websocket_connection.send(json.dumps(signal_data))
                response = await self.websocket_connection.recv()
                self.logger.info(f"Signal sent via WebSocket: {response}")
            else:
                # Fallback to HTTP
                await self._send_http_signal(signal_data)
        except Exception as e:
            self.logger.error(f"Failed to send signal: {e}")
            await self._send_http_signal(signal_data)

    async def _send_http_signal(self, signal_data: Dict):
        """Send signal via HTTP API"""
        import requests
        try:
            response = requests.post(
                f"{self.http_api_url}/api/ml_signal",
                json=signal_data,
                timeout=10
            )
            if response.status_code == 200:
                self.logger.info("Signal sent via HTTP API")
            else:
                self.logger.error(f"HTTP API error: {response.status_code}")
        except Exception as e:
            self.logger.error(f"HTTP API failed: {e}")

    async def fetch_market_data(self, symbol: str) -> Dict:
        """Fetch current market data from main system"""
        import requests
        try:
            response = requests.get(
                f"{self.http_api_url}/api/market_data",
                params={'symbol': symbol},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Market data fetch failed: {response.status_code}")
                return {}
        except Exception as e:
            self.logger.error(f"Market data fetch error: {e}")
            return {}

    async def close_connection(self):
        """Close WebSocket connection"""
        if self.websocket_connection:
            await self.websocket_connection.close()
            self.logger.info("WebSocket connection closed")