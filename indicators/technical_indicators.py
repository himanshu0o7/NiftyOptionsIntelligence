import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
# import talib  # TA-Lib not available, using manual implementations
from utils.logger import Logger

class TechnicalIndicators:
    """Technical indicators for trading analysis"""

    def __init__(self):
        self.logger = Logger()

    def calculate_rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            if len(data) < period:
                return pd.Series(dtype=float)

            # Manual RSI calculation
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(dtype=float)

    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        try:
            if len(data) < period:
                return pd.Series(dtype=float)

            # Manual EMA calculation
            return data.ewm(span=period, adjust=False).mean()

        except Exception as e:
            self.logger.error(f"Error calculating EMA: {str(e)}")
            return pd.Series(dtype=float)

    def calculate_sma(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        try:
            if len(data) < period:
                return pd.Series(dtype=float)

            return data.rolling(window=period).mean()

        except Exception as e:
            self.logger.error(f"Error calculating SMA: {str(e)}")
            return pd.Series(dtype=float)

    def calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        try:
            if data.empty or len(data) == 0:
                return pd.Series(dtype=float)

            # Extract required columns with fallbacks
            high = data['high'] if 'high' in data.columns else data['close']
            low = data['low'] if 'low' in data.columns else data['close']
            close = data['close']
            volume = data['volume'] if 'volume' in data.columns else pd.Series([1000] * len(data))

            # Typical price (HLC/3)
            typical_price = (high + low + close) / 3

            # VWAP calculation
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            return vwap

        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {str(e)}")
            return pd.Series(dtype=float)

    def calculate_bollinger_bands(self, data: pd.Series, period: int = 20,
                                 std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            if len(data) < period:
                return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

            # Manual calculation
            middle = data.rolling(window=period).mean()
            std = data.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return upper, middle, lower

        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    def calculate_macd(self, data: pd.Series, fast_period: int = 12,
                      slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if len(data) < slow_period:
                return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

            # Manual calculation
            ema_fast = self.calculate_ema(data, fast_period)
            ema_slow = self.calculate_ema(data, slow_period)
            macd_line = ema_fast - ema_slow
            signal_line = self.calculate_ema(macd_line, signal_period)
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram

        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                           k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        try:
            if len(high) < k_period:
                return pd.Series(dtype=float), pd.Series(dtype=float)

            # Manual calculation
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            return k_percent, d_percent

        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {str(e)}")
            return pd.Series(dtype=float), pd.Series(dtype=float)

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            if len(high) < period:
                return pd.Series(dtype=float)

            # Manual calculation
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr

        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(dtype=float)

    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        try:
            if len(high) < period:
                return pd.Series(dtype=float)

            # Manual ADX calculation (simplified)
            plus_dm = high.diff()
            minus_dm = low.diff() * -1

            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            tr = self.calculate_atr(high, low, close, 1)
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            return adx

        except Exception as e:
            self.logger.error(f"Error calculating ADX: {str(e)}")
            return pd.Series(dtype=float)

    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series,
                           period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        try:
            if len(high) < period:
                return pd.Series(dtype=float)

            # Manual calculation
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            willr = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return willr

        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {str(e)}")
            return pd.Series(dtype=float)

    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        try:
            if len(high) < period:
                return pd.Series(dtype=float)

            # Manual calculation
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - np.mean(x)))
            )
            cci = (typical_price - sma_tp) / (0.015 * mad)
            return cci

        except Exception as e:
            self.logger.error(f"Error calculating CCI: {str(e)}")
            return pd.Series(dtype=float)

    def identify_support_resistance(self, data: pd.Series, window: int = 20) -> Dict[str, List[float]]:
        """Identify support and resistance levels"""
        try:
            if len(data) < window * 2:
                return {'support': [], 'resistance': []}

            # Find local minima (support) and maxima (resistance)
            support_levels = []
            resistance_levels = []

            for i in range(window, len(data) - window):
                # Check for local minimum (support)
                if data.iloc[i] == data.iloc[i-window:i+window+1].min():
                    support_levels.append(data.iloc[i])

                # Check for local maximum (resistance)
                if data.iloc[i] == data.iloc[i-window:i+window+1].max():
                    resistance_levels.append(data.iloc[i])

            # Remove duplicates and sort
            support_levels = sorted(list(set(support_levels)))
            resistance_levels = sorted(list(set(resistance_levels)), reverse=True)

            return {
                'support': support_levels[-5:],  # Last 5 support levels
                'resistance': resistance_levels[:5]  # Top 5 resistance levels
            }

        except Exception as e:
            self.logger.error(f"Error identifying support/resistance: {str(e)}")
            return {'support': [], 'resistance': []}

    def detect_breakout(self, current_price: float, resistance_level: float,
                       threshold_pct: float = 1.0) -> bool:
        """Detect breakout above resistance"""
        try:
            breakout_price = resistance_level * (1 + threshold_pct / 100)
            return current_price > breakout_price
        except Exception as e:
            self.logger.error(f"Error detecting breakout: {str(e)}")
            return False

    def detect_breakdown(self, current_price: float, support_level: float,
                        threshold_pct: float = 1.0) -> bool:
        """Detect breakdown below support"""
        try:
            breakdown_price = support_level * (1 - threshold_pct / 100)
            return current_price < breakdown_price
        except Exception as e:
            self.logger.error(f"Error detecting breakdown: {str(e)}")
            return False

    def calculate_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate pivot points for support and resistance"""
        try:
            pivot = (high + low + close) / 3

            # Resistance levels
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)

            # Support levels
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)

            return {
                'pivot': round(pivot, 2),
                'r1': round(r1, 2),
                'r2': round(r2, 2),
                'r3': round(r3, 2),
                's1': round(s1, 2),
                's2': round(s2, 2),
                's3': round(s3, 2)
            }

        except Exception as e:
            self.logger.error(f"Error calculating pivot points: {str(e)}")
            return {}

    def calculate_fibonacci_retracement(self, high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            diff = high - low

            levels = {
                '0%': high,
                '23.6%': high - 0.236 * diff,
                '38.2%': high - 0.382 * diff,
                '50%': high - 0.5 * diff,
                '61.8%': high - 0.618 * diff,
                '78.6%': high - 0.786 * diff,
                '100%': low
            }

            return {k: round(v, 2) for k, v in levels.items()}

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci retracement: {str(e)}")
            return {}

    def get_trend_direction(self, short_ema: pd.Series, long_ema: pd.Series) -> str:
        """Determine trend direction based on EMA crossover"""
        try:
            if len(short_ema) == 0 or len(long_ema) == 0:
                return 'NEUTRAL'

            current_short = short_ema.iloc[-1]
            current_long = long_ema.iloc[-1]

            if current_short > current_long:
                return 'BULLISH'
            elif current_short < current_long:
                return 'BEARISH'
            else:
                return 'NEUTRAL'

        except Exception as e:
            self.logger.error(f"Error determining trend direction: {str(e)}")
            return 'NEUTRAL'

    def calculate_momentum(self, data: pd.Series, period: int = 10) -> pd.Series:
        """Calculate price momentum"""
        try:
            if len(data) < period:
                return pd.Series(dtype=float)

            momentum = data / data.shift(period) - 1
            return momentum * 100  # Convert to percentage

        except Exception as e:
            self.logger.error(f"Error calculating momentum: {str(e)}")
            return pd.Series(dtype=float)

    def calculate_rate_of_change(self, data: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Rate of Change (ROC)"""
        try:
            if len(data) < period:
                return pd.Series(dtype=float)

            # Manual calculation
            roc = ((data - data.shift(period)) / data.shift(period)) * 100
            return roc

        except Exception as e:
            self.logger.error(f"Error calculating ROC: {str(e)}")
            return pd.Series(dtype=float)

    def get_technical_summary(self, data: pd.DataFrame) -> Dict[str, any]:
        """Get comprehensive technical analysis summary"""
        try:
            if len(data) < 20:
                return {}

            close = data['close_price']
            high = data['high_price']
            low = data['low_price']
            volume = data['volume']

            # Calculate key indicators
            rsi = self.calculate_rsi(close)
            ema_9 = self.calculate_ema(close, 9)
            ema_21 = self.calculate_ema(close, 21)
            macd_line, signal_line, histogram = self.calculate_macd(close)
            upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(close)

            current_price = close.iloc[-1]

            summary = {
                'current_price': current_price,
                'rsi': rsi.iloc[-1] if len(rsi) > 0 else None,
                'rsi_signal': self._get_rsi_signal(rsi.iloc[-1] if len(rsi) > 0 else 50),
                'ema_trend': self.get_trend_direction(ema_9, ema_21),
                'macd_signal': 'BULLISH' if len(histogram) > 0 and histogram.iloc[-1] > 0 else 'BEARISH',
                'bb_position': self._get_bb_position(current_price, upper_bb.iloc[-1] if len(upper_bb) > 0 else 0,
                                                   lower_bb.iloc[-1] if len(lower_bb) > 0 else 0),
                'volume_trend': 'HIGH' if len(volume) > 0 and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] else 'NORMAL',
                'support_resistance': self.identify_support_resistance(close),
                'overall_signal': 'NEUTRAL'  # Will be determined based on other signals
            }

            # Determine overall signal
            bullish_signals = 0
            bearish_signals = 0

            if summary['rsi_signal'] == 'BULLISH':
                bullish_signals += 1
            elif summary['rsi_signal'] == 'BEARISH':
                bearish_signals += 1

            if summary['ema_trend'] == 'BULLISH':
                bullish_signals += 1
            elif summary['ema_trend'] == 'BEARISH':
                bearish_signals += 1

            if summary['macd_signal'] == 'BULLISH':
                bullish_signals += 1
            else:
                bearish_signals += 1

            if bullish_signals > bearish_signals:
                summary['overall_signal'] = 'BULLISH'
            elif bearish_signals > bullish_signals:
                summary['overall_signal'] = 'BEARISH'

            return summary

        except Exception as e:
            self.logger.error(f"Error generating technical summary: {str(e)}")
            return {}

    def _get_rsi_signal(self, rsi_value: float) -> str:
        """Get RSI signal interpretation"""
        if rsi_value > 70:
            return 'OVERBOUGHT'
        elif rsi_value < 30:
            return 'OVERSOLD'
        elif rsi_value > 50:
            return 'BULLISH'
        else:
            return 'BEARISH'

    def _get_bb_position(self, current_price: float, upper_band: float, lower_band: float) -> str:
        """Get Bollinger Band position"""
        if current_price > upper_band:
            return 'ABOVE_UPPER'
        elif current_price < lower_band:
            return 'BELOW_LOWER'
        else:
            return 'WITHIN_BANDS'
