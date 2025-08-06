import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from strategies.base_strategy import BaseStrategy
from indicators.technical_indicators import TechnicalIndicators

class OIAnalysis(BaseStrategy):
    """Open Interest analysis strategy for options trading"""

    def __init__(self, config: Dict):
        super().__init__("OI Analysis", config)

        # OI Analysis parameters
        self.oi_change_threshold = config.get('oi_change_threshold', 20.0)  # Percentage
        self.volume_oi_ratio_threshold = config.get('volume_oi_ratio_threshold', 0.3)
        self.max_pain_deviation = config.get('max_pain_deviation', 2.0)  # Percentage
        self.pcr_threshold = config.get('pcr_threshold', {'low': 0.7, 'high': 1.3})

        # Data storage for OI analysis
        self.oi_data = {}
        self.max_pain_levels = {}
        self.pcr_data = {}
        self.oi_buildup_signals = []

        # Technical indicators
        self.indicators = TechnicalIndicators()

        self.logger.info(f"Initialized {self.name} with OI threshold: {self.oi_change_threshold}%")

    def analyze(self, market_data: Dict) -> List[Dict]:
        """Analyze Open Interest data for trading signals"""
        signals = []

        # Group data by underlying
        underlying_data = self._group_by_underlying(market_data)

        for underlying, options_data in underlying_data.items():
            # Update OI data
            self._update_oi_data(underlying, options_data)

            # Calculate max pain
            max_pain = self._calculate_max_pain(underlying, options_data)
            if max_pain:
                self.max_pain_levels[underlying] = max_pain

            # Calculate Put-Call Ratio
            pcr = self._calculate_pcr(underlying, options_data)
            if pcr:
                self.pcr_data[underlying] = pcr

            # Generate signals based on OI analysis
            oi_signals = self._analyze_oi_buildup(underlying, options_data)
            signals.extend(oi_signals)

            # Generate signals based on max pain deviation
            max_pain_signals = self._analyze_max_pain_deviation(underlying, options_data)
            signals.extend(max_pain_signals)

            # Generate signals based on PCR
            pcr_signals = self._analyze_pcr(underlying, options_data)
            signals.extend(pcr_signals)

        return signals

    def should_enter(self, symbol: str, data: Dict) -> Optional[Dict]:
        """Determine entry based on OI analysis"""
        if not self.check_risk_limits():
            return None

        underlying = self._extract_underlying(symbol)
        if not underlying:
            return None

        # Get OI data for the symbol
        oi_change = self._get_oi_change(symbol, data)
        volume_oi_ratio = self._get_volume_oi_ratio(symbol, data)

        # Check for significant OI buildup
        if abs(oi_change) >= self.oi_change_threshold:
            signal_type = 'OI_BUILDUP_BULLISH' if oi_change > 0 else 'OI_BUILDUP_BEARISH'
            action = self._determine_action_from_oi(symbol, data, oi_change)

            if action:
                confidence = self._calculate_oi_confidence(symbol, data, oi_change, volume_oi_ratio)

                return self.generate_signal(
                    symbol=symbol,
                    action=action,
                    price=data.get('ltp', 0),
                    signal_type=signal_type,
                    confidence=confidence,
                    parameters={
                        'oi_change_percentage': oi_change,
                        'volume_oi_ratio': volume_oi_ratio,
                        'current_oi': data.get('open_interest', 0),
                        'underlying': underlying
                    }
                )

        return None

    def should_exit(self, symbol: str, position: Dict, current_data: Dict) -> Optional[Dict]:
        """Determine exit based on OI analysis and standard risk management"""
        current_price = current_data.get('ltp', 0)

        # Standard risk management checks
        if self.apply_stop_loss(position, current_price):
            return self.generate_signal(
                symbol=symbol,
                action='SELL' if position['transaction_type'] == 'BUY' else 'BUY',
                price=current_price,
                signal_type='STOP_LOSS',
                confidence=1.0,
                parameters={'reason': 'stop_loss_triggered'}
            )

        if self.apply_take_profit(position, current_price):
            return self.generate_signal(
                symbol=symbol,
                action='SELL' if position['transaction_type'] == 'BUY' else 'BUY',
                price=current_price,
                signal_type='TAKE_PROFIT',
                confidence=1.0,
                parameters={'reason': 'take_profit_triggered'}
            )

        # OI-based exit signals
        oi_exit_signal = self._check_oi_exit_conditions(symbol, position, current_data)
        if oi_exit_signal:
            return oi_exit_signal

        return None

    def _group_by_underlying(self, market_data: Dict) -> Dict[str, Dict]:
        """Group options data by underlying asset"""
        underlying_groups = {}

        for symbol, data in market_data.items():
            underlying = self._extract_underlying(symbol)
            if underlying:
                if underlying not in underlying_groups:
                    underlying_groups[underlying] = {}
                underlying_groups[underlying][symbol] = data

        return underlying_groups

    def _extract_underlying(self, symbol: str) -> Optional[str]:
        """Extract underlying asset from option symbol"""
        if 'NIFTY' in symbol and 'BANKNIFTY' not in symbol:
            return 'NIFTY'
        elif 'BANKNIFTY' in symbol:
            return 'BANKNIFTY'
        return None

    def _update_oi_data(self, underlying: str, options_data: Dict):
        """Update OI data for analysis"""
        if underlying not in self.oi_data:
            self.oi_data[underlying] = {}

        current_time = datetime.now()

        for symbol, data in options_data.items():
            oi = data.get('open_interest', 0)

            if symbol not in self.oi_data[underlying]:
                self.oi_data[underlying][symbol] = []

            # Store OI data with timestamp
            self.oi_data[underlying][symbol].append({
                'timestamp': current_time,
                'oi': oi,
                'volume': data.get('volume', 0),
                'ltp': data.get('ltp', 0)
            })

            # Keep only recent data (last 24 hours)
            cutoff_time = current_time - timedelta(hours=24)
            self.oi_data[underlying][symbol] = [
                entry for entry in self.oi_data[underlying][symbol]
                if entry['timestamp'] >= cutoff_time
            ]

    def _calculate_max_pain(self, underlying: str, options_data: Dict) -> Optional[float]:
        """Calculate max pain level for the underlying"""
        try:
            total_pain = {}

            for symbol, data in options_data.items():
                # Extract strike price and option type
                strike_price, option_type = self._parse_option_symbol(symbol)
                if not strike_price or not option_type:
                    continue

                oi = data.get('open_interest', 0)
                if oi == 0:
                    continue

                # Calculate pain for different price levels around current strikes
                price_range = range(int(strike_price * 0.9), int(strike_price * 1.1), 50)

                for price in price_range:
                    if price not in total_pain:
                        total_pain[price] = 0

                    # Calculate pain for option writers
                    if option_type == 'CE' and price > strike_price:
                        pain = (price - strike_price) * oi
                    elif option_type == 'PE' and price < strike_price:
                        pain = (strike_price - price) * oi
                    else:
                        pain = 0

                    total_pain[price] += pain

            # Find price level with minimum pain (max pain)
            if total_pain:
                max_pain_price = min(total_pain.keys(), key=lambda k: total_pain[k])
                return float(max_pain_price)

            return None

        except Exception as e:
            self.logger.error(f"Error calculating max pain for {underlying}: {str(e)}")
            return None

    def _calculate_pcr(self, underlying: str, options_data: Dict) -> Optional[Dict]:
        """Calculate Put-Call Ratio based on OI and volume"""
        try:
            put_oi = 0
            call_oi = 0
            put_volume = 0
            call_volume = 0

            for symbol, data in options_data.items():
                _, option_type = self._parse_option_symbol(symbol)
                if not option_type:
                    continue

                oi = data.get('open_interest', 0)
                volume = data.get('volume', 0)

                if option_type == 'PE':
                    put_oi += oi
                    put_volume += volume
                elif option_type == 'CE':
                    call_oi += oi
                    call_volume += volume

            pcr_oi = put_oi / call_oi if call_oi > 0 else 0
            pcr_volume = put_volume / call_volume if call_volume > 0 else 0

            return {
                'pcr_oi': pcr_oi,
                'pcr_volume': pcr_volume,
                'put_oi': put_oi,
                'call_oi': call_oi,
                'put_volume': put_volume,
                'call_volume': call_volume,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Error calculating PCR for {underlying}: {str(e)}")
            return None

    def _analyze_oi_buildup(self, underlying: str, options_data: Dict) -> List[Dict]:
        """Analyze OI buildup patterns"""
        signals = []

        try:
            for symbol, data in options_data.items():
                oi_change = self._get_oi_change(symbol, data)

                if abs(oi_change) >= self.oi_change_threshold:
                    strike_price, option_type = self._parse_option_symbol(symbol)
                    if not strike_price or not option_type:
                        continue

                    # Determine signal based on OI buildup and option type
                    signal_type, action = self._interpret_oi_buildup(option_type, oi_change, data)

                    if signal_type and action:
                        confidence = self._calculate_oi_confidence(symbol, data, oi_change,
                                                                 self._get_volume_oi_ratio(symbol, data))

                        signal = self.generate_signal(
                            symbol=symbol,
                            action=action,
                            price=data.get('ltp', 0),
                            signal_type=signal_type,
                            confidence=confidence,
                            parameters={
                                'oi_change_percentage': oi_change,
                                'strike_price': strike_price,
                                'option_type': option_type,
                                'underlying': underlying,
                                'interpretation': self._get_oi_interpretation(option_type, oi_change)
                            }
                        )

                        signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"Error analyzing OI buildup for {underlying}: {str(e)}")
            return []

    def _analyze_max_pain_deviation(self, underlying: str, options_data: Dict) -> List[Dict]:
        """Analyze signals based on max pain deviation"""
        signals = []

        try:
            max_pain = self.max_pain_levels.get(underlying)
            if not max_pain:
                return signals

            # Get current underlying price (approximate from ATM options)
            current_price = self._estimate_underlying_price(options_data)
            if not current_price:
                return signals

            # Calculate deviation from max pain
            deviation_pct = abs((current_price - max_pain) / max_pain) * 100

            if deviation_pct >= self.max_pain_deviation:
                # Price is significantly away from max pain - expect mean reversion
                action = 'SELL' if current_price > max_pain else 'BUY'
                signal_type = 'MAX_PAIN_REVERSION'

                # Find suitable options for the signal
                suitable_options = self._find_max_pain_options(underlying, options_data, max_pain, action)

                for symbol in suitable_options:
                    if symbol in options_data:
                        signal = self.generate_signal(
                            symbol=symbol,
                            action=action,
                            price=options_data[symbol].get('ltp', 0),
                            signal_type=signal_type,
                            confidence=0.6,
                            parameters={
                                'max_pain': max_pain,
                                'current_price': current_price,
                                'deviation_percentage': deviation_pct,
                                'underlying': underlying
                            }
                        )
                        signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"Error analyzing max pain deviation for {underlying}: {str(e)}")
            return []

    def _analyze_pcr(self, underlying: str, options_data: Dict) -> List[Dict]:
        """Analyze signals based on Put-Call Ratio"""
        signals = []

        try:
            pcr_data = self.pcr_data.get(underlying)
            if not pcr_data:
                return signals

            pcr_oi = pcr_data['pcr_oi']
            pcr_volume = pcr_data['pcr_volume']

            # Extreme PCR values indicate potential reversals
            if pcr_oi < self.pcr_threshold['low']:
                # Very low PCR - excessive bullishness, potential bearish reversal
                signal_type = 'PCR_BEARISH_REVERSAL'
                action = 'SELL'
                confidence = 0.7

            elif pcr_oi > self.pcr_threshold['high']:
                # Very high PCR - excessive bearishness, potential bullish reversal
                signal_type = 'PCR_BULLISH_REVERSAL'
                action = 'BUY'
                confidence = 0.7

            else:
                return signals

            # Find suitable options for PCR-based signals
            suitable_options = self._find_pcr_options(underlying, options_data, signal_type)

            for symbol in suitable_options:
                if symbol in options_data:
                    signal = self.generate_signal(
                        symbol=symbol,
                        action=action,
                        price=options_data[symbol].get('ltp', 0),
                        signal_type=signal_type,
                        confidence=confidence,
                        parameters={
                            'pcr_oi': pcr_oi,
                            'pcr_volume': pcr_volume,
                            'threshold_low': self.pcr_threshold['low'],
                            'threshold_high': self.pcr_threshold['high'],
                            'underlying': underlying
                        }
                    )
                    signals.append(signal)

            return signals

        except Exception as e:
            self.logger.error(f"Error analyzing PCR for {underlying}: {str(e)}")
            return []

    def _get_oi_change(self, symbol: str, current_data: Dict) -> float:
        """Calculate OI change percentage"""
        try:
            underlying = self._extract_underlying(symbol)
            if not underlying or underlying not in self.oi_data:
                return 0.0

            if symbol not in self.oi_data[underlying]:
                return 0.0

            oi_history = self.oi_data[underlying][symbol]
            if len(oi_history) < 2:
                return 0.0

            current_oi = current_data.get('open_interest', 0)
            previous_oi = oi_history[-2]['oi'] if len(oi_history) >= 2 else oi_history[-1]['oi']

            if previous_oi == 0:
                return 0.0

            oi_change = ((current_oi - previous_oi) / previous_oi) * 100
            return round(oi_change, 2)

        except Exception as e:
            self.logger.error(f"Error calculating OI change for {symbol}: {str(e)}")
            return 0.0

    def _get_volume_oi_ratio(self, symbol: str, data: Dict) -> float:
        """Calculate Volume to OI ratio"""
        try:
            volume = data.get('volume', 0)
            oi = data.get('open_interest', 0)

            if oi == 0:
                return 0.0

            return round(volume / oi, 3)

        except Exception as e:
            self.logger.error(f"Error calculating Volume/OI ratio for {symbol}: {str(e)}")
            return 0.0

    def _parse_option_symbol(self, symbol: str) -> Tuple[Optional[float], Optional[str]]:
        """Parse option symbol to extract strike price and type"""
        try:
            # Example: NIFTY25JAN24C21500
            if 'CE' in symbol or 'C' in symbol[-7:]:
                option_type = 'CE'
                # Extract strike price (last 5 digits typically)
                import re
                match = re.search(r'C(\d+)$', symbol)
                if match:
                    strike_price = float(match.group(1))
                else:
                    return None, None
            elif 'PE' in symbol or 'P' in symbol[-7:]:
                option_type = 'PE'
                import re
                match = re.search(r'P(\d+)$', symbol)
                if match:
                    strike_price = float(match.group(1))
                else:
                    return None, None
            else:
                return None, None

            return strike_price, option_type

        except Exception as e:
            self.logger.error(f"Error parsing option symbol {symbol}: {str(e)}")
            return None, None

    def _interpret_oi_buildup(self, option_type: str, oi_change: float, data: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Interpret OI buildup to determine signal type and action"""
        try:
            volume_oi_ratio = self._get_volume_oi_ratio('', data)  # Symbol not needed for this calculation

            if oi_change > 0:  # OI increasing
                if option_type == 'CE':
                    if volume_oi_ratio > self.volume_oi_ratio_threshold:
                        # High volume with OI buildup in calls - bearish (selling calls)
                        return 'OI_CALL_WRITING', 'SELL'
                    else:
                        # Low volume with OI buildup in calls - bullish (buying calls)
                        return 'OI_CALL_BUYING', 'BUY'

                elif option_type == 'PE':
                    if volume_oi_ratio > self.volume_oi_ratio_threshold:
                        # High volume with OI buildup in puts - bullish (selling puts)
                        return 'OI_PUT_WRITING', 'SELL'
                    else:
                        # Low volume with OI buildup in puts - bearish (buying puts)
                        return 'OI_PUT_BUYING', 'BUY'

            return None, None

        except Exception as e:
            self.logger.error(f"Error interpreting OI buildup: {str(e)}")
            return None, None

    def _get_oi_interpretation(self, option_type: str, oi_change: float) -> str:
        """Get human-readable interpretation of OI change"""
        if oi_change > 0:
            if option_type == 'CE':
                return "Call OI buildup - potential resistance or bullish coverage"
            else:
                return "Put OI buildup - potential support or bearish coverage"
        else:
            if option_type == 'CE':
                return "Call OI reduction - covering or profit booking"
            else:
                return "Put OI reduction - covering or profit booking"

    def _calculate_oi_confidence(self, symbol: str, data: Dict, oi_change: float, volume_oi_ratio: float) -> float:
        """Calculate confidence score for OI-based signals"""
        try:
            base_confidence = 0.5

            # Higher OI change increases confidence
            oi_factor = min(0.3, abs(oi_change) / 100)
            base_confidence += oi_factor

            # Volume/OI ratio confirmation
            if volume_oi_ratio > self.volume_oi_ratio_threshold:
                base_confidence += 0.15

            # Additional factors can be added here

            return min(0.9, base_confidence)

        except Exception as e:
            self.logger.error(f"Error calculating OI confidence for {symbol}: {str(e)}")
            return 0.5

    def _check_oi_exit_conditions(self, symbol: str, position: Dict, current_data: Dict) -> Optional[Dict]:
        """Check OI-based exit conditions"""
        try:
            # Check if OI is reducing significantly (unwinding)
            oi_change = self._get_oi_change(symbol, current_data)

            if abs(oi_change) >= self.oi_change_threshold and oi_change < 0:
                # Significant OI reduction - consider exit
                return self.generate_signal(
                    symbol=symbol,
                    action='SELL' if position['transaction_type'] == 'BUY' else 'BUY',
                    price=current_data.get('ltp', 0),
                    signal_type='OI_UNWINDING',
                    confidence=0.8,
                    parameters={
                        'reason': 'oi_unwinding',
                        'oi_change_percentage': oi_change
                    }
                )

            return None

        except Exception as e:
            self.logger.error(f"Error checking OI exit conditions for {symbol}: {str(e)}")
            return None

    def _estimate_underlying_price(self, options_data: Dict) -> Optional[float]:
        """Estimate underlying price from ATM options"""
        try:
            # Find options closest to ATM
            strikes_and_prices = []

            for symbol, data in options_data.items():
                strike_price, option_type = self._parse_option_symbol(symbol)
                if strike_price and option_type == 'CE':  # Use call options
                    ltp = data.get('ltp', 0)
                    if ltp > 0:
                        # Approximate underlying price as strike + call premium
                        estimated_price = strike_price + ltp
                        strikes_and_prices.append((strike_price, estimated_price))

            if strikes_and_prices:
                # Return the median estimate
                estimates = [price for _, price in strikes_and_prices]
                return np.median(estimates)

            return None

        except Exception as e:
            self.logger.error(f"Error estimating underlying price: {str(e)}")
            return None

    def _find_max_pain_options(self, underlying: str, options_data: Dict, max_pain: float, action: str) -> List[str]:
        """Find suitable options for max pain reversion signals"""
        suitable_options = []

        try:
            for symbol, data in options_data.items():
                strike_price, option_type = self._parse_option_symbol(symbol)
                if not strike_price or not option_type:
                    continue

                # Select options based on max pain level and action
                if action == 'BUY' and option_type == 'CE' and strike_price <= max_pain * 1.02:
                    suitable_options.append(symbol)
                elif action == 'SELL' and option_type == 'PE' and strike_price >= max_pain * 0.98:
                    suitable_options.append(symbol)

            return suitable_options[:3]  # Limit to top 3 options

        except Exception as e:
            self.logger.error(f"Error finding max pain options: {str(e)}")
            return []

    def _find_pcr_options(self, underlying: str, options_data: Dict, signal_type: str) -> List[str]:
        """Find suitable options for PCR-based signals"""
        suitable_options = []

        try:
            for symbol, data in options_data.items():
                strike_price, option_type = self._parse_option_symbol(symbol)
                if not strike_price or not option_type:
                    continue

                # Select appropriate options based on PCR signal
                if signal_type == 'PCR_BULLISH_REVERSAL' and option_type == 'CE':
                    suitable_options.append(symbol)
                elif signal_type == 'PCR_BEARISH_REVERSAL' and option_type == 'PE':
                    suitable_options.append(symbol)

            return suitable_options[:3]  # Limit to top 3 options

        except Exception as e:
            self.logger.error(f"Error finding PCR options: {str(e)}")
            return []

    def _determine_action_from_oi(self, symbol: str, data: Dict, oi_change: float) -> Optional[str]:
        """Determine trading action based on OI analysis"""
        try:
            _, option_type = self._parse_option_symbol(symbol)
            if not option_type:
                return None

            volume_oi_ratio = self._get_volume_oi_ratio(symbol, data)

            # Use OI interpretation logic
            _, action = self._interpret_oi_buildup(option_type, oi_change, data)
            return action

        except Exception as e:
            self.logger.error(f"Error determining action from OI for {symbol}: {str(e)}")
            return None

    def get_oi_summary(self, underlying: str) -> Dict:
        """Get OI analysis summary for an underlying"""
        try:
            summary = {
                'underlying': underlying,
                'max_pain': self.max_pain_levels.get(underlying),
                'pcr_data': self.pcr_data.get(underlying),
                'total_call_oi': 0,
                'total_put_oi': 0,
                'significant_oi_changes': [],
                'updated_at': datetime.now()
            }

            if underlying in self.oi_data:
                for symbol, oi_history in self.oi_data[underlying].items():
                    if oi_history:
                        latest_oi = oi_history[-1]['oi']
                        _, option_type = self._parse_option_symbol(symbol)

                        if option_type == 'CE':
                            summary['total_call_oi'] += latest_oi
                        elif option_type == 'PE':
                            summary['total_put_oi'] += latest_oi

            return summary

        except Exception as e:
            self.logger.error(f"Error generating OI summary for {underlying}: {str(e)}")
            return {}
