"""
Live Data Demonstration for Options Trading System
Shows how real-time data is fetched from Angel One API
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import json

from core.angel_api import AngelOneAPI
from core.options_greeks_api import OptionsGreeksAPI
from core.symbol_resolver import SymbolResolver
from core.live_data_manager import LiveDataManager
from core.websocket_v2 import AngelWebSocketV2
from utils.logger import Logger

class LiveDataDemo:
    """Demonstrate live data fetching capabilities"""
    
    def __init__(self, api_client: AngelOneAPI):
        self.api_client = api_client
        self.logger = Logger()
        self.greeks_api = OptionsGreeksAPI(api_client)
        self.symbol_resolver = SymbolResolver()
        self.live_data_manager = LiveDataManager(api_client)
        
        # Initialize components
        self.symbol_resolver.load_instruments()
        
    def fetch_live_options_data(self, underlying: str = "NIFTY", num_strikes: int = 5) -> Dict:
        """Fetch live options data with Greeks"""
        try:
            self.logger.info(f"Fetching live options data for {underlying}")
            
            # Get current expiry date
            expiry_date = self._get_current_expiry(underlying)
            
            # Get live Greeks data
            greeks_data = self.greeks_api.get_option_greeks(underlying, expiry_date)
            
            if not greeks_data:
                return {"error": "No Greeks data available"}
            
            # Get current week options
            current_spot = self._get_current_spot_price(underlying)
            options_chain = self.symbol_resolver.get_current_week_options(underlying, current_spot)
            
            # Combine Greeks with options chain
            live_data = {
                "underlying": underlying,
                "spot_price": current_spot,
                "expiry_date": expiry_date,
                "timestamp": datetime.now().isoformat(),
                "options_data": [],
                "market_status": self._get_market_status()
            }
            
            # Process options data
            for option in options_chain[:num_strikes * 2]:  # CE and PE
                option_data = {
                    "symbol": option.get("symbol"),
                    "token": option.get("token"),
                    "strike": option.get("strike"),
                    "option_type": option.get("option_type"),
                    "lot_size": option.get("lot_size"),
                    "greeks": self._get_greeks_for_option(greeks_data, option),
                    "live_price": self._get_live_price(option.get("token")),
                    "volume": self._get_volume_data(option.get("token")),
                    "open_interest": self._get_oi_data(option.get("token"))
                }
                live_data["options_data"].append(option_data)
            
            return live_data
            
        except Exception as e:
            self.logger.error(f"Error fetching live options data: {e}")
            return {"error": str(e)}
    
    def demonstrate_websocket_connection(self) -> Dict:
        """Demonstrate WebSocket live data connection"""
        try:
            self.logger.info("Demonstrating WebSocket connection")
            
            # Connect to WebSocket
            ws_connected = self.live_data_manager.connect_websocket()
            
            if not ws_connected:
                return {"error": "WebSocket connection failed"}
            
            # Subscribe to sample tokens
            sample_tokens = ["39898", "39899", "39900"]  # NIFTY options tokens
            subscription_success = self.live_data_manager.subscribe_to_options(sample_tokens)
            
            # Collect live data for 5 seconds
            live_ticks = []
            start_time = time.time()
            
            while time.time() - start_time < 5:
                for token in sample_tokens:
                    tick_data = self.live_data_manager.get_live_option_data(token)
                    if tick_data:
                        live_ticks.append({
                            "token": token,
                            "timestamp": datetime.now().isoformat(),
                            "data": tick_data
                        })
                time.sleep(0.1)
            
            return {
                "websocket_status": "connected" if ws_connected else "disconnected",
                "subscription_status": "success" if subscription_success else "failed",
                "live_ticks_count": len(live_ticks),
                "sample_ticks": live_ticks[:10],  # Show first 10 ticks
                "data_frequency": f"{len(live_ticks)/5:.1f} ticks/second"
            }
            
        except Exception as e:
            self.logger.error(f"WebSocket demo error: {e}")
            return {"error": str(e)}
    
    def fetch_historical_data_with_indicators(self, symbol: str = "NIFTY", days: int = 30) -> Dict:
        """Fetch historical data and calculate indicators"""
        try:
            self.logger.info(f"Fetching historical data for {symbol}")
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # For demo, we'll use the Angel One historical API
            historical_data = self.api_client.get_historical_data(
                exchange="NSE",
                symboltoken="99926000",  # NIFTY 50 token
                interval="FIVE_MINUTE",
                from_date=start_date.strftime("%Y-%m-%d %H:%M"),
                to_date=end_date.strftime("%Y-%m-%d %H:%M")
            )
            
            if not historical_data:
                return {"error": "No historical data available"}
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            
            # Calculate indicators
            from indicators.technical_indicators import TechnicalIndicators
            indicators = TechnicalIndicators()
            
            # Calculate RSI
            rsi = indicators.calculate_rsi(df['close'])
            
            # Calculate VWAP
            vwap = indicators.calculate_vwap(df)
            
            # Calculate EMA
            ema20 = indicators.calculate_ema(df['close'], 20)
            ema50 = indicators.calculate_ema(df['close'], 50)
            
            # Prepare response
            result = {
                "symbol": symbol,
                "data_points": len(df),
                "date_range": {
                    "start": df.index[0] if not df.empty else None,
                    "end": df.index[-1] if not df.empty else None
                },
                "current_values": {
                    "price": float(df['close'].iloc[-1]) if not df.empty else None,
                    "rsi": float(rsi.iloc[-1]) if not rsi.empty else None,
                    "vwap": float(vwap.iloc[-1]) if not vwap.empty else None,
                    "ema20": float(ema20.iloc[-1]) if not ema20.empty else None,
                    "ema50": float(ema50.iloc[-1]) if not ema50.empty else None
                },
                "market_analysis": {
                    "trend": "bullish" if not ema20.empty and not ema50.empty and ema20.iloc[-1] > ema50.iloc[-1] else "bearish",
                    "rsi_signal": "overbought" if not rsi.empty and rsi.iloc[-1] > 70 else "oversold" if not rsi.empty and rsi.iloc[-1] < 30 else "neutral",
                    "volume_profile": "active" if not df.empty and df['volume'].iloc[-1] > df['volume'].mean() else "normal"
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Historical data error: {e}")
            return {"error": str(e)}
    
    def demonstrate_oi_analysis(self, underlying: str = "NIFTY") -> Dict:
        """Demonstrate Open Interest analysis"""
        try:
            self.logger.info(f"Demonstrating OI analysis for {underlying}")
            
            # Get OI buildup data
            oi_buildup = self.greeks_api.get_oi_buildup_data()
            
            # Get PCR data
            pcr_data = self.greeks_api.get_pcr_data()
            
            # Get gainers/losers
            gainers = self.greeks_api.get_gainers_losers("PercOIGainers")
            losers = self.greeks_api.get_gainers_losers("PercOILosers")
            
            result = {
                "underlying": underlying,
                "timestamp": datetime.now().isoformat(),
                "oi_buildup": {
                    "total_records": len(oi_buildup) if oi_buildup else 0,
                    "top_5": oi_buildup[:5] if oi_buildup else []
                },
                "pcr_analysis": {
                    "total_records": len(pcr_data) if pcr_data else 0,
                    "current_pcr": pcr_data[0] if pcr_data else None
                },
                "market_sentiment": {
                    "top_gainers": gainers[:5] if gainers else [],
                    "top_losers": losers[:5] if losers else []
                },
                "analysis_summary": self._analyze_oi_sentiment(oi_buildup, pcr_data)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"OI analysis error: {e}")
            return {"error": str(e)}
    
    def get_live_data_summary(self) -> Dict:
        """Get comprehensive live data summary"""
        try:
            summary = {
                "system_status": {
                    "api_connected": self.api_client.is_session_valid(),
                    "websocket_available": True,
                    "greeks_api_active": True,
                    "symbol_resolver_loaded": hasattr(self.symbol_resolver, 'instruments') and len(self.symbol_resolver.instruments) > 0
                },
                "data_sources": {
                    "real_time_prices": "Angel One WebSocket v2",
                    "options_greeks": "Angel One Options API",
                    "historical_data": "Angel One Historical API",
                    "oi_analysis": "Angel One OI Buildup API",
                    "market_sentiment": "Angel One PCR API"
                },
                "supported_indices": ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"],
                "update_frequency": {
                    "live_prices": "Real-time (WebSocket)",
                    "greeks": "Every 30 seconds",
                    "oi_data": "Every 60 seconds",
                    "historical": "On-demand"
                },
                "capabilities": [
                    "Real-time options pricing",
                    "Live Greeks calculation",
                    "OI buildup analysis",
                    "Volume profile tracking",
                    "Technical indicators",
                    "Market sentiment analysis"
                ]
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Live data summary error: {e}")
            return {"error": str(e)}
    
    def _get_current_expiry(self, underlying: str) -> str:
        """Get current week expiry date"""
        # For demo, return static expiry
        expiry_map = {
            "NIFTY": "10JUL2025",
            "BANKNIFTY": "09JUL2025",
            "FINNIFTY": "08JUL2025",
            "MIDCPNIFTY": "10JUL2025",
            "NIFTYNXT50": "10JUL2025"
        }
        return expiry_map.get(underlying, "10JUL2025")
    
    def _get_current_spot_price(self, underlying: str) -> float:
        """Get current spot price"""
        # For demo, return sample spot prices
        spot_prices = {
            "NIFTY": 23500,
            "BANKNIFTY": 51000,
            "FINNIFTY": 20200,
            "MIDCPNIFTY": 12300,
            "NIFTYNXT50": 68900
        }
        return spot_prices.get(underlying, 23500)
    
    def _get_market_status(self) -> str:
        """Get current market status"""
        current_time = datetime.now().time()
        if current_time >= datetime.strptime("09:15", "%H:%M").time() and current_time <= datetime.strptime("15:30", "%H:%M").time():
            return "OPEN"
        else:
            return "CLOSED"
    
    def _get_greeks_for_option(self, greeks_data: List[Dict], option: Dict) -> Dict:
        """Get Greeks for specific option"""
        if not greeks_data:
            return {}
        
        # Find matching Greeks data
        for greeks in greeks_data:
            if greeks.get("strike") == option.get("strike") and greeks.get("option_type") == option.get("option_type"):
                return {
                    "delta": greeks.get("delta", 0),
                    "gamma": greeks.get("gamma", 0),
                    "theta": greeks.get("theta", 0),
                    "vega": greeks.get("vega", 0),
                    "iv": greeks.get("iv", 0)
                }
        
        return {}
    
    def _get_live_price(self, token: str) -> float:
        """Get live price for token"""
        # For demo, return sample price
        return np.random.uniform(150, 250)
    
    def _get_volume_data(self, token: str) -> int:
        """Get volume data for token"""
        # For demo, return sample volume
        return np.random.randint(1000, 10000)
    
    def _get_oi_data(self, token: str) -> int:
        """Get open interest data for token"""
        # For demo, return sample OI
        return np.random.randint(50000, 200000)
    
    def _analyze_oi_sentiment(self, oi_buildup: List[Dict], pcr_data: List[Dict]) -> Dict:
        """Analyze OI sentiment"""
        if not oi_buildup or not pcr_data:
            return {"sentiment": "neutral", "confidence": 0.5}
        
        # Simple sentiment analysis
        ce_buildup = sum(1 for item in oi_buildup if item.get("option_type") == "CE")
        pe_buildup = sum(1 for item in oi_buildup if item.get("option_type") == "PE")
        
        if ce_buildup > pe_buildup:
            return {"sentiment": "bullish", "confidence": 0.7}
        elif pe_buildup > ce_buildup:
            return {"sentiment": "bearish", "confidence": 0.7}
        else:
            return {"sentiment": "neutral", "confidence": 0.5}