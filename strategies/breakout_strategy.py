import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from strategies.base_strategy import BaseStrategy
from indicators.technical_indicators import TechnicalIndicators

class BreakoutStrategy(BaseStrategy):
    """Breakout trading strategy for options"""
    
    def __init__(self, config: Dict):
        super().__init__("Breakout Strategy", config)
        
        # Strategy-specific parameters
        self.breakout_threshold = config.get('breakout_threshold', 2.0)  # Percentage
        self.volume_multiplier = config.get('volume_multiplier', 1.5)
        self.confirmation_candles = config.get('confirmation_candles', 2)
        self.lookback_period = config.get('lookback_period', 20)
        
        # Technical indicators
        self.indicators = TechnicalIndicators()
        
        # Breakout tracking
        self.resistance_levels = {}
        self.support_levels = {}
        self.breakout_signals = []
        
        self.logger.info(f"Initialized {self.name} with threshold: {self.breakout_threshold}%")
    
    def analyze(self, market_data: Dict) -> List[Dict]:
        """Analyze market data for breakout opportunities"""
        signals = []
        
        for symbol, data in market_data.items():
            if self._is_eligible_symbol(symbol):
                # Get historical data
                historical_data = self._get_historical_data(symbol)
                
                if historical_data is not None and len(historical_data) >= self.lookback_period:
                    # Identify support and resistance levels
                    self._identify_levels(symbol, historical_data)
                    
                    # Check for breakout signals
                    breakout_signal = self._check_breakout(symbol, data, historical_data)
                    if breakout_signal:
                        signals.append(breakout_signal)
                    
                    # Check for breakdown signals
                    breakdown_signal = self._check_breakdown(symbol, data, historical_data)
                    if breakdown_signal:
                        signals.append(breakdown_signal)
        
        return signals
    
    def should_enter(self, symbol: str, data: Dict) -> Optional[Dict]:
        """Determine if strategy should enter a position"""
        if not self.check_risk_limits():
            return None
        
        current_price = data.get('ltp', 0)
        volume = data.get('volume', 0)
        
        # Check if symbol has breakout levels identified
        if symbol not in self.resistance_levels and symbol not in self.support_levels:
            return None
        
        # Get historical data for confirmation
        historical_data = self._get_historical_data(symbol)
        if historical_data is None:
            return None
        
        # Calculate average volume
        avg_volume = historical_data['volume'].rolling(window=20).mean().iloc[-1]
        
        # Check breakout conditions
        resistance = self.resistance_levels.get(symbol, float('inf'))
        support = self.support_levels.get(symbol, 0)
        
        # Bullish breakout
        if current_price > resistance and volume > avg_volume * self.volume_multiplier:
            return self.generate_signal(
                symbol=symbol,
                action='BUY',
                price=current_price,
                signal_type='BREAKOUT',
                confidence=self._calculate_confidence(symbol, data, 'bullish'),
                parameters={
                    'resistance_level': resistance,
                    'volume_ratio': volume / avg_volume,
                    'breakout_percentage': ((current_price - resistance) / resistance) * 100
                }
            )
        
        # Bearish breakdown
        if current_price < support and volume > avg_volume * self.volume_multiplier:
            return self.generate_signal(
                symbol=symbol,
                action='SELL',
                price=current_price,
                signal_type='BREAKDOWN',
                confidence=self._calculate_confidence(symbol, data, 'bearish'),
                parameters={
                    'support_level': support,
                    'volume_ratio': volume / avg_volume,
                    'breakdown_percentage': ((support - current_price) / support) * 100
                }
            )
        
        return None
    
    def should_exit(self, symbol: str, position: Dict, current_data: Dict) -> Optional[Dict]:
        """Determine if strategy should exit a position"""
        current_price = current_data.get('ltp', 0)
        
        # Apply stop loss
        if self.apply_stop_loss(position, current_price):
            return self.generate_signal(
                symbol=symbol,
                action='SELL' if position['transaction_type'] == 'BUY' else 'BUY',
                price=current_price,
                signal_type='STOP_LOSS',
                confidence=1.0,
                parameters={'reason': 'stop_loss_triggered'}
            )
        
        # Apply take profit
        if self.apply_take_profit(position, current_price):
            return self.generate_signal(
                symbol=symbol,
                action='SELL' if position['transaction_type'] == 'BUY' else 'BUY',
                price=current_price,
                signal_type='TAKE_PROFIT',
                confidence=1.0,
                parameters={'reason': 'take_profit_triggered'}
            )
        
        # Apply trailing stop
        if self.apply_trailing_stop(position, current_price):
            return self.generate_signal(
                symbol=symbol,
                action='SELL' if position['transaction_type'] == 'BUY' else 'BUY',
                price=current_price,
                signal_type='TRAILING_STOP',
                confidence=1.0,
                parameters={'reason': 'trailing_stop_triggered'}
            )
        
        # Check for reversal signals
        reversal_signal = self._check_reversal(symbol, position, current_data)
        if reversal_signal:
            return reversal_signal
        
        return None
    
    def _identify_levels(self, symbol: str, data: pd.DataFrame):
        """Identify support and resistance levels"""
        try:
            # Calculate pivot points
            highs = data['high_price'].rolling(window=5, center=True).max()
            lows = data['low_price'].rolling(window=5, center=True).min()
            
            # Identify resistance levels (pivot highs)
            resistance_points = []
            for i in range(2, len(data) - 2):
                if (data['high_price'].iloc[i] == highs.iloc[i] and 
                    data['high_price'].iloc[i] > data['high_price'].iloc[i-1] and
                    data['high_price'].iloc[i] > data['high_price'].iloc[i+1]):
                    resistance_points.append(data['high_price'].iloc[i])
            
            # Identify support levels (pivot lows)
            support_points = []
            for i in range(2, len(data) - 2):
                if (data['low_price'].iloc[i] == lows.iloc[i] and 
                    data['low_price'].iloc[i] < data['low_price'].iloc[i-1] and
                    data['low_price'].iloc[i] < data['low_price'].iloc[i+1]):
                    support_points.append(data['low_price'].iloc[i])
            
            # Store the most recent and significant levels
            if resistance_points:
                self.resistance_levels[symbol] = max(resistance_points[-3:])  # Recent high
            
            if support_points:
                self.support_levels[symbol] = min(support_points[-3:])  # Recent low
                
        except Exception as e:
            self.logger.error(f"Error identifying levels for {symbol}: {str(e)}")
    
    def _check_breakout(self, symbol: str, current_data: Dict, historical_data: pd.DataFrame) -> Optional[Dict]:
        """Check for bullish breakout"""
        try:
            current_price = current_data.get('ltp', 0)
            volume = current_data.get('volume', 0)
            
            resistance = self.resistance_levels.get(symbol)
            if not resistance:
                return None
            
            # Calculate breakout percentage
            breakout_pct = ((current_price - resistance) / resistance) * 100
            
            # Check if price has broken above resistance with sufficient momentum
            if breakout_pct >= self.breakout_threshold:
                # Confirm with volume
                avg_volume = historical_data['volume'].rolling(window=20).mean().iloc[-1]
                volume_confirmation = volume > avg_volume * self.volume_multiplier
                
                # Additional technical confirmation
                rsi = self.indicators.calculate_rsi(historical_data['close_price'])
                if len(rsi) > 0:
                    current_rsi = rsi.iloc[-1]
                    rsi_confirmation = 50 < current_rsi < 80  # Bullish but not overbought
                else:
                    rsi_confirmation = True
                
                if volume_confirmation and rsi_confirmation:
                    confidence = min(0.9, 0.5 + (breakout_pct / 10) + (0.1 if volume_confirmation else 0))
                    
                    return self.generate_signal(
                        symbol=symbol,
                        action='BUY',
                        price=current_price,
                        signal_type='BREAKOUT',
                        confidence=confidence,
                        parameters={
                            'resistance_level': resistance,
                            'breakout_percentage': breakout_pct,
                            'volume_ratio': volume / avg_volume,
                            'rsi': current_rsi if len(rsi) > 0 else None
                        }
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking breakout for {symbol}: {str(e)}")
            return None
    
    def _check_breakdown(self, symbol: str, current_data: Dict, historical_data: pd.DataFrame) -> Optional[Dict]:
        """Check for bearish breakdown"""
        try:
            current_price = current_data.get('ltp', 0)
            volume = current_data.get('volume', 0)
            
            support = self.support_levels.get(symbol)
            if not support:
                return None
            
            # Calculate breakdown percentage
            breakdown_pct = ((support - current_price) / support) * 100
            
            # Check if price has broken below support with sufficient momentum
            if breakdown_pct >= self.breakout_threshold:
                # Confirm with volume
                avg_volume = historical_data['volume'].rolling(window=20).mean().iloc[-1]
                volume_confirmation = volume > avg_volume * self.volume_multiplier
                
                # Additional technical confirmation
                rsi = self.indicators.calculate_rsi(historical_data['close_price'])
                if len(rsi) > 0:
                    current_rsi = rsi.iloc[-1]
                    rsi_confirmation = 20 < current_rsi < 50  # Bearish but not oversold
                else:
                    rsi_confirmation = True
                
                if volume_confirmation and rsi_confirmation:
                    confidence = min(0.9, 0.5 + (breakdown_pct / 10) + (0.1 if volume_confirmation else 0))
                    
                    return self.generate_signal(
                        symbol=symbol,
                        action='SELL',
                        price=current_price,
                        signal_type='BREAKDOWN',
                        confidence=confidence,
                        parameters={
                            'support_level': support,
                            'breakdown_percentage': breakdown_pct,
                            'volume_ratio': volume / avg_volume,
                            'rsi': current_rsi if len(rsi) > 0 else None
                        }
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking breakdown for {symbol}: {str(e)}")
            return None
    
    def _check_reversal(self, symbol: str, position: Dict, current_data: Dict) -> Optional[Dict]:
        """Check for reversal patterns"""
        try:
            historical_data = self._get_historical_data(symbol)
            if historical_data is None or len(historical_data) < 10:
                return None
            
            current_price = current_data.get('ltp', 0)
            
            # Calculate RSI for reversal confirmation
            rsi = self.indicators.calculate_rsi(historical_data['close_price'])
            if len(rsi) == 0:
                return None
            
            current_rsi = rsi.iloc[-1]
            
            # For long positions, check for bearish reversal
            if position['transaction_type'] == 'BUY':
                if current_rsi > 80:  # Overbought
                    # Check for bearish divergence or reversal pattern
                    recent_highs = historical_data['high_price'].tail(5)
                    if current_price < recent_highs.max() * 0.98:  # 2% below recent high
                        return self.generate_signal(
                            symbol=symbol,
                            action='SELL',
                            price=current_price,
                            signal_type='REVERSAL',
                            confidence=0.7,
                            parameters={
                                'reason': 'overbought_reversal',
                                'rsi': current_rsi
                            }
                        )
            
            # For short positions, check for bullish reversal
            elif position['transaction_type'] == 'SELL':
                if current_rsi < 20:  # Oversold
                    # Check for bullish divergence or reversal pattern
                    recent_lows = historical_data['low_price'].tail(5)
                    if current_price > recent_lows.min() * 1.02:  # 2% above recent low
                        return self.generate_signal(
                            symbol=symbol,
                            action='BUY',
                            price=current_price,
                            signal_type='REVERSAL',
                            confidence=0.7,
                            parameters={
                                'reason': 'oversold_reversal',
                                'rsi': current_rsi
                            }
                        )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking reversal for {symbol}: {str(e)}")
            return None
    
    def _calculate_confidence(self, symbol: str, data: Dict, direction: str) -> float:
        """Calculate signal confidence based on multiple factors"""
        try:
            base_confidence = 0.5
            
            # Volume confirmation
            volume = data.get('volume', 0)
            historical_data = self._get_historical_data(symbol)
            if historical_data is not None:
                avg_volume = historical_data['volume'].rolling(window=20).mean().iloc[-1]
                volume_ratio = volume / avg_volume
                
                if volume_ratio > self.volume_multiplier:
                    base_confidence += 0.2
                elif volume_ratio > 1.0:
                    base_confidence += 0.1
            
            # Price momentum
            current_price = data.get('ltp', 0)
            if direction == 'bullish':
                resistance = self.resistance_levels.get(symbol, current_price)
                momentum = ((current_price - resistance) / resistance) * 100
            else:
                support = self.support_levels.get(symbol, current_price)
                momentum = ((support - current_price) / support) * 100
            
            if momentum > self.breakout_threshold * 2:
                base_confidence += 0.2
            elif momentum > self.breakout_threshold:
                base_confidence += 0.1
            
            return min(0.95, base_confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence for {symbol}: {str(e)}")
            return 0.5
    
    def _get_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get historical data for analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            return self.db.get_market_data(symbol, start_date, end_date)
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None
    
    def _is_eligible_symbol(self, symbol: str) -> bool:
        """Check if symbol is eligible for breakout strategy"""
        # Focus on NIFTY and BANKNIFTY options
        eligible_underlyings = ['NIFTY', 'BANKNIFTY']
        
        for underlying in eligible_underlyings:
            if underlying in symbol:
                return True
        
        return False
    
    def get_breakout_levels(self, symbol: str) -> Dict:
        """Get current support and resistance levels for a symbol"""
        return {
            'symbol': symbol,
            'resistance': self.resistance_levels.get(symbol),
            'support': self.support_levels.get(symbol),
            'updated_at': datetime.now()
        }
    
    def update_levels(self, symbol: str, resistance: float = None, support: float = None):
        """Manually update support/resistance levels"""
        if resistance is not None:
            self.resistance_levels[symbol] = resistance
            
        if support is not None:
            self.support_levels[symbol] = support
            
        self.logger.info(f"Updated levels for {symbol}: R={resistance}, S={support}")

# strategies/breakout_strategy.py

from strategies.base import BaseStrategy

class BreakoutStrategy(BaseStrategy):
    def should_enter(self):
        return self.data["volume"] > 100000  # Example logic

    def should_exit(self):
        return self.data["volume"] < 50000  # Example logic

def should_enter(self, option_data):
    delta = option_data.get("delta", 0)
    volume = option_data.get("volume", 0)
    vega = option_data.get("vega", 0)

    if delta > 0.5 and volume > 2000 and vega > 1.0:
        return True
    return False
