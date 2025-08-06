import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import ta
from utils.logger import Logger

class FeatureEngineer:
    """Advanced feature engineering for ML-based signal generation"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.logger = Logger()
        self.feature_names = []

    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical analysis features"""
        try:
            features_df = df.copy()

            # Price-based features
            features_df['price_change'] = features_df['close'].pct_change()
            features_df['high_low_ratio'] = features_df['high'] / features_df['low']
            features_df['close_open_ratio'] = features_df['close'] / features_df['open']

            # Volume features
            features_df['volume_sma'] = features_df['volume'].rolling(20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
            features_df['price_volume'] = features_df['close'] * features_df['volume']

            # Volatility features
            features_df['volatility'] = features_df['price_change'].rolling(20).std()
            features_df['atr'] = ta.volatility.average_true_range(
                features_df['high'], features_df['low'], features_df['close'], window=14
            )

            # Momentum indicators
            features_df['rsi'] = ta.momentum.rsi(features_df['close'], window=14)
            features_df['stoch'] = ta.momentum.stoch(
                features_df['high'], features_df['low'], features_df['close']
            )
            features_df['williams_r'] = ta.momentum.williams_r(
                features_df['high'], features_df['low'], features_df['close']
            )

            # Trend indicators
            features_df['sma_5'] = ta.trend.sma_indicator(features_df['close'], window=5)
            features_df['sma_20'] = ta.trend.sma_indicator(features_df['close'], window=20)
            features_df['ema_12'] = ta.trend.ema_indicator(features_df['close'], window=12)
            features_df['ema_26'] = ta.trend.ema_indicator(features_df['close'], window=26)

            # MACD
            features_df['macd'] = ta.trend.macd(features_df['close'])
            features_df['macd_signal'] = ta.trend.macd_signal(features_df['close'])
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']

            # Bollinger Bands
            features_df['bb_upper'] = ta.volatility.bollinger_hband(features_df['close'])
            features_df['bb_lower'] = ta.volatility.bollinger_lband(features_df['close'])
            features_df['bb_middle'] = ta.volatility.bollinger_mavg(features_df['close'])
            features_df['bb_width'] = features_df['bb_upper'] - features_df['bb_lower']
            features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / features_df['bb_width']

            # Support and Resistance levels
            features_df['resistance'] = features_df['high'].rolling(20).max()
            features_df['support'] = features_df['low'].rolling(20).min()
            features_df['resistance_distance'] = (features_df['resistance'] - features_df['close']) / features_df['close']
            features_df['support_distance'] = (features_df['close'] - features_df['support']) / features_df['close']

            return features_df

        except Exception as e:
            self.logger.error(f"Feature engineering error: {str(e)}")
            return df

    def create_options_features(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """Create features specific to options trading"""
        try:
            features_df = options_data.copy()

            # Open Interest features
            features_df['oi_change'] = features_df['open_interest'].pct_change()
            features_df['oi_volume_ratio'] = features_df['open_interest'] / features_df['volume']

            # Put-Call Ratio
            if 'option_type' in features_df.columns:
                calls = features_df[features_df['option_type'] == 'CE']
                puts = features_df[features_df['option_type'] == 'PE']

                if not calls.empty and not puts.empty:
                    pcr_volume = puts['volume'].sum() / calls['volume'].sum()
                    pcr_oi = puts['open_interest'].sum() / calls['open_interest'].sum()

                    features_df['pcr_volume'] = pcr_volume
                    features_df['pcr_oi'] = pcr_oi

            # Greeks-based features
            if 'delta' in features_df.columns:
                features_df['delta_change'] = features_df['delta'].pct_change()
                features_df['gamma_delta_ratio'] = features_df['gamma'] / abs(features_df['delta'])
                features_df['theta_vega_ratio'] = features_df['theta'] / features_df['vega']

            # Implied Volatility features
            if 'iv' in features_df.columns:
                features_df['iv_rank'] = features_df['iv'].rolling(252).rank(pct=True)
                features_df['iv_change'] = features_df['iv'].pct_change()

            return features_df

        except Exception as e:
            self.logger.error(f"Options feature engineering error: {str(e)}")
            return options_data

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        try:
            features_df = df.copy()

            # Ensure datetime index
            if not isinstance(features_df.index, pd.DatetimeIndex):
                features_df.index = pd.to_datetime(features_df.index)

            # Time features
            features_df['hour'] = features_df.index.hour
            features_df['day_of_week'] = features_df.index.dayofweek
            features_df['day_of_month'] = features_df.index.day
            features_df['month'] = features_df.index.month
            features_df['quarter'] = features_df.index.quarter

            # Market session features
            features_df['is_opening_hour'] = (features_df['hour'] == 9).astype(int)
            features_df['is_closing_hour'] = (features_df['hour'] == 15).astype(int)
            features_df['is_lunch_time'] = ((features_df['hour'] >= 12) & (features_df['hour'] <= 13)).astype(int)

            # Weekly patterns
            features_df['is_monday'] = (features_df['day_of_week'] == 0).astype(int)
            features_df['is_friday'] = (features_df['day_of_week'] == 4).astype(int)
            features_df['is_weekend_effect'] = ((features_df['day_of_week'] == 0) | (features_df['day_of_week'] == 4)).astype(int)

            # Expiry-related features for options
            features_df['days_to_expiry'] = 0  # Would calculate based on expiry date
            features_df['is_expiry_week'] = 0  # Would mark expiry week

            return features_df

        except Exception as e:
            self.logger.error(f"Time feature engineering error: {str(e)}")
            return df

    def create_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Create lagged features for time series"""
        try:
            features_df = df.copy()

            for col in columns:
                if col in features_df.columns:
                    for lag in lags:
                        features_df[f'{col}_lag_{lag}'] = features_df[col].shift(lag)

            return features_df

        except Exception as e:
            self.logger.error(f"Lag feature engineering error: {str(e)}")
            return df

    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """Create rolling window statistical features"""
        try:
            features_df = df.copy()

            for col in columns:
                if col in features_df.columns:
                    for window in windows:
                        features_df[f'{col}_mean_{window}'] = features_df[col].rolling(window).mean()
                        features_df[f'{col}_std_{window}'] = features_df[col].rolling(window).std()
                        features_df[f'{col}_min_{window}'] = features_df[col].rolling(window).min()
                        features_df[f'{col}_max_{window}'] = features_df[col].rolling(window).max()
                        features_df[f'{col}_skew_{window}'] = features_df[col].rolling(window).skew()

            return features_df

        except Exception as e:
            self.logger.error(f"Rolling feature engineering error: {str(e)}")
            return df

    def prepare_features(self, df: pd.DataFrame, options_data: pd.DataFrame = None) -> pd.DataFrame:
        """Prepare comprehensive feature set for ML models"""
        try:
            # Create technical features
            features_df = self.create_technical_features(df)

            # Add time features
            features_df = self.create_time_features(features_df)

            # Add lag features
            price_cols = ['close', 'volume', 'rsi', 'macd']
            features_df = self.create_lag_features(features_df, price_cols, [1, 2, 3, 5])

            # Add rolling features
            features_df = self.create_rolling_features(features_df, ['close', 'volume'], [5, 10, 20])

            # Add options features if provided
            if options_data is not None:
                options_features = self.create_options_features(options_data)
                # Merge with main features (implementation depends on data structure)

            # Remove infinite and NaN values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(0)

            # Store feature names
            self.feature_names = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]

            self.logger.info(f"Created {len(self.feature_names)} features for ML models")
            return features_df

        except Exception as e:
            self.logger.error(f"Feature preparation error: {str(e)}")
            return df

    def scale_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features for ML models"""
        try:
            feature_cols = [col for col in features.columns if col in self.feature_names]

            if fit:
                scaled_features = self.scaler.fit_transform(features[feature_cols])
            else:
                scaled_features = self.scaler.transform(features[feature_cols])

            scaled_df = features.copy()
            scaled_df[feature_cols] = scaled_features

            return scaled_df

        except Exception as e:
            self.logger.error(f"Feature scaling error: {str(e)}")
            return features

    def reduce_dimensions(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction"""
        try:
            feature_cols = [col for col in features.columns if col in self.feature_names]

            if fit:
                reduced_features = self.pca.fit_transform(features[feature_cols])
            else:
                reduced_features = self.pca.transform(features[feature_cols])

            # Create new DataFrame with PCA components
            pca_df = pd.DataFrame(
                reduced_features,
                index=features.index,
                columns=[f'pca_{i}' for i in range(reduced_features.shape[1])]
            )

            # Add non-feature columns back
            non_feature_cols = [col for col in features.columns if col not in self.feature_names]
            for col in non_feature_cols:
                pca_df[col] = features[col].values

            self.logger.info(f"Reduced features from {len(feature_cols)} to {reduced_features.shape[1]} components")
            return pca_df

        except Exception as e:
            self.logger.error(f"PCA dimensionality reduction error: {str(e)}")
            return features