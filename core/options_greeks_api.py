"""
Angel One Options Greeks and OI Analysis API
"""
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from utils.logger import Logger

class OptionsGreeksAPI:
    """Angel One Options Greeks and Market Data API"""

    def __init__(self, api_client):
        self.api_client = api_client
        self.logger = Logger()
        self.base_url = "https://apiconnect.angelone.in"

    def get_option_greeks(self, underlying: str, expiry_date: str) -> Optional[List[Dict]]:
        """Get option Greeks for underlying and expiry"""
        try:
            url = f"{self.base_url}/rest/secure/angelbroking/marketData/v1/optionGreek"

            # Convert expiry date format from "10JUL25" to "10JUL2025"
            if len(expiry_date) == 7:  # "10JUL25" format
                day_month = expiry_date[:5]  # "10JUL"
                year = "20" + expiry_date[5:]  # "2025"
                formatted_expiry = day_month + year
            else:
                formatted_expiry = expiry_date

            payload = {
                "name": underlying,
                "expirydate": formatted_expiry
            }

            headers = self._get_headers()

            self.logger.info(f"Greeks API request: {underlying} expiry {formatted_expiry}")

            response = requests.post(url, json=payload, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    self.logger.info(f"Retrieved {len(data.get('data', []))} Greeks for {underlying}")
                    return data.get('data', [])
                else:
                    self.logger.error(f"Greeks API error: {data.get('message')}")
                    return None
            else:
                self.logger.error(f"Greeks API HTTP error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching Greeks: {e}")
            return None

    def get_oi_buildup_data(self, expiry_type: str = "NEAR", data_type: str = "Long Built Up") -> Optional[List[Dict]]:
        """Get OI buildup data"""
        try:
            url = f"{self.base_url}/rest/secure/angelbroking/marketData/v1/OIBuildup"

            payload = {
                "expirytype": expiry_type,
                "datatype": data_type
            }

            headers = self._get_headers()

            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    return data.get('data', [])
                else:
                    self.logger.error(f"OI Buildup API error: {data.get('message')}")
                    return None
            else:
                self.logger.error(f"OI Buildup API HTTP error: {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching OI buildup: {e}")
            return None

    def get_gainers_losers(self, data_type: str = "PercOIGainers", expiry_type: str = "NEAR") -> Optional[List[Dict]]:
        """Get top gainers/losers with OI data"""
        try:
            url = f"{self.base_url}/rest/secure/angelbroking/marketData/v1/gainersLosers"

            payload = {
                "datatype": data_type,
                "expirytype": expiry_type
            }

            headers = self._get_headers()

            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    return data.get('data', [])
                else:
                    self.logger.error(f"Gainers/Losers API error: {data.get('message')}")
                    return None
            else:
                self.logger.error(f"Gainers/Losers API HTTP error: {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching gainers/losers: {e}")
            return None

    def get_pcr_data(self) -> Optional[List[Dict]]:
        """Get Put-Call Ratio data"""
        try:
            url = f"{self.base_url}/rest/secure/angelbroking/marketData/v1/putCallRatio"

            headers = self._get_headers()

            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    return data.get('data', [])
                else:
                    self.logger.error(f"PCR API error: {data.get('message')}")
                    return None
            else:
                self.logger.error(f"PCR API HTTP error: {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching PCR: {e}")
            return None

    def get_historical_oi_data(self, symbol_token: str, from_date: str, to_date: str,
                              interval: str = "FIFTEEN_MINUTE") -> Optional[List[Dict]]:
        """Get historical OI data"""
        try:
            url = f"{self.base_url}/rest/secure/angelbroking/historical/v1/getOIData"

            payload = {
                "exchange": "NFO",
                "symboltoken": symbol_token,
                "interval": interval,
                "fromdate": from_date,
                "todate": to_date
            }

            headers = self._get_headers()

            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                data = response.json()
                if data.get('status'):
                    return data.get('data', [])
                else:
                    self.logger.error(f"Historical OI API error: {data.get('message')}")
                    return None
            else:
                self.logger.error(f"Historical OI API HTTP error: {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching historical OI: {e}")
            return None

    def analyze_option_for_trading(self, underlying: str, expiry_date: str,
                                  strike_price: float, option_type: str) -> Dict:
        """Comprehensive option analysis for trading decision"""
        try:
            # Get Greeks data
            greeks_data = self.get_option_greeks(underlying, expiry_date)

            # Get OI buildup data
            oi_buildup = self.get_oi_buildup_data()

            # Get PCR data
            pcr_data = self.get_pcr_data()

            # Find specific option data
            option_data = None
            if greeks_data:
                for option in greeks_data:
                    if (float(option.get('strikePrice', 0)) == strike_price and
                        option.get('optionType') == option_type):
                        option_data = option
                        break

            analysis = {
                'strike': strike_price,
                'option_type': option_type,
                'underlying': underlying,
                'expiry': expiry_date,
                'timestamp': datetime.now().isoformat(),
                'trade_recommendation': 'HOLD'
            }

            if option_data:
                # Greeks analysis
                delta = float(option_data.get('delta', 0))
                gamma = float(option_data.get('gamma', 0))
                theta = float(option_data.get('theta', 0))
                vega = float(option_data.get('vega', 0))
                iv = float(option_data.get('impliedVolatility', 0))
                volume = float(option_data.get('tradeVolume', 0))

                analysis.update({
                    'delta': delta,
                    'gamma': gamma,
                    'theta': theta,
                    'vega': vega,
                    'implied_volatility': iv,
                    'trade_volume': volume,
                    'iv_percentile': self._calculate_iv_percentile(iv),
                    'liquidity_score': self._calculate_liquidity_score(volume),
                    'time_decay_risk': abs(theta),
                    'volatility_sensitivity': abs(vega)
                })

                # Trading recommendation based on Greeks
                if option_type == 'CE':
                    if delta > 0.4 and gamma > 0.001 and iv < 25:
                        analysis['trade_recommendation'] = 'BUY'
                        analysis['confidence'] = 0.8
                    elif delta < 0.2 or iv > 35:
                        analysis['trade_recommendation'] = 'AVOID'
                        analysis['confidence'] = 0.3
                else:  # PE
                    if delta < -0.4 and gamma > 0.001 and iv < 25:
                        analysis['trade_recommendation'] = 'BUY'
                        analysis['confidence'] = 0.8
                    elif delta > -0.2 or iv > 35:
                        analysis['trade_recommendation'] = 'AVOID'
                        analysis['confidence'] = 0.3

            # Add market sentiment from PCR
            if pcr_data:
                nifty_pcr = next((item for item in pcr_data if 'NIFTY' in item.get('tradingSymbol', '')), None)
                if nifty_pcr:
                    pcr_value = nifty_pcr.get('pcr', 1.0)
                    analysis['pcr'] = pcr_value
                    analysis['market_sentiment'] = 'BULLISH' if pcr_value < 0.8 else 'BEARISH' if pcr_value > 1.2 else 'NEUTRAL'

            return analysis

        except Exception as e:
            self.logger.error(f"Error in option analysis: {e}")
            return {'error': str(e)}

    def _get_headers(self) -> Dict:
        """Get API headers"""
        if not self.api_client or not hasattr(self.api_client, 'jwt_token'):
            raise Exception("API client not connected")

        return {
            'Authorization': f'Bearer {self.api_client.jwt_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-UserType': 'USER',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': self.api_client.client_local_ip,
            'X-ClientPublicIP': self.api_client.client_public_ip,
            'X-MACAddress': self.api_client.mac_address,
            'X-PrivateKey': self.api_client.api_key
        }

    def _calculate_iv_percentile(self, iv: float) -> int:
        """Calculate IV percentile (simplified)"""
        # This would normally require historical IV data
        # For now, using a simple classification
        if iv < 15:
            return 20
        elif iv < 25:
            return 50
        else:
            return 80

    def _calculate_liquidity_score(self, volume: float) -> float:
        """Calculate liquidity score based on volume"""
        if volume > 50000:
            return 0.9
        elif volume > 10000:
            return 0.7
        elif volume > 1000:
            return 0.5
        else:
            return 0.2