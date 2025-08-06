#!/usr/bin/env python3
"""
Independent ML Trading Bot
A separate machine learning module that can be run independently
and communicate with the main trading system via API calls
"""

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import time
import asyncio
import websockets
from dataclasses import dataclass

# ML imports
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_bot.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class MLSignal:
    """ML Signal data structure"""
    symbol: str
    action: str  # BUY_CE, BUY_PE, WAIT
    confidence: float
    prediction_source: str
    features_used: List[str]
    timestamp: str
    reasoning: str
    target_strike: float
    expected_move: float

class MLTradingBot:
    """Independent ML Trading Bot"""

    def __init__(self, config_path: str = "ml_bot/config.json"):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.models = {}
        self.ensemble_model = None
        self.is_trained = False

        # WebSocket connection for real-time communication
        self.websocket_url = self.config.get('websocket_url', 'ws://localhost:8765')
        self.main_system_api = self.config.get('main_system_api', 'http://localhost:5000')

        # ML Configuration
        self.feature_columns = [
            'rsi', 'ema_short', 'ema_long', 'vwap_ratio',
            'volume_ratio', 'price_momentum', 'volatility',
            'delta_change', 'gamma_exposure', 'iv_percentile'
        ]

        self.logger.info("ML Trading Bot initialized")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration
            default_config = {
                "model_params": {
                    "random_forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
                    "svm": {"C": 1.0, "kernel": "rbf", "probability": True, "random_state": 42},
                    "logistic": {"C": 1.0, "random_state": 42, "max_iter": 1000},
                    "neural_network": {"hidden_layer_sizes": (100, 50), "max_iter": 1000, "random_state": 42}
                },
                "training_params": {
                    "test_size": 0.2,
                    "min_confidence": 0.65,
                    "retrain_interval_hours": 24
                },
                "websocket_url": "ws://localhost:8765",
                "main_system_api": "http://localhost:5000"
            }

            # Save default config
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)

            return default_config

    def initialize_models(self):
        """Initialize ML models"""
        try:
            model_params = self.config['model_params']

            # Random Forest
            self.models['random_forest'] = RandomForestClassifier(
                **model_params['random_forest']
            )

            # SVM
            self.models['svm'] = SVC(
                **model_params['svm']
            )

            # Logistic Regression
            self.models['logistic'] = LogisticRegression(
                **model_params['logistic']
            )

            # Neural Network
            self.models['neural_network'] = MLPClassifier(
                **model_params['neural_network']
            )

            # Ensemble Model
            self.ensemble_model = VotingClassifier(
                estimators=[
                    ('rf', self.models['random_forest']),
                    ('svm', self.models['svm']),
                    ('lr', self.models['logistic']),
                    ('nn', self.models['neural_network'])
                ],
                voting='soft'
            )

            self.logger.info("ML models initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise

    def fetch_market_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Fetch market data from main system"""
        try:
            # Call main system API to get market data
            url = f"{self.main_system_api}/api/market_data"
            params = {
                'symbol': symbol,
                'days': days,
                'include_options': True
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['market_data'])
                return df
            else:
                self.logger.error(f"Failed to fetch market data: {response.status_code}")
                return self._generate_sample_data(symbol, days)

        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return self._generate_sample_data(symbol, days)

    def _generate_sample_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate sample market data for testing"""
        np.random.seed(42)
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')

        # Generate realistic market data
        base_price = 23500 if symbol == 'NIFTY' else 50000
        price_changes = np.random.normal(0, 0.02, days)
        prices = [base_price]

        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))

        data = {
            'date': dates,
            'close': prices,
            'volume': np.random.normal(1000000, 200000, days),
            'rsi': np.random.uniform(30, 70, days),
            'ema_short': [p * 0.99 for p in prices],
            'ema_long': [p * 0.98 for p in prices],
            'vwap_ratio': np.random.uniform(0.98, 1.02, days),
            'volume_ratio': np.random.uniform(0.5, 2.0, days),
            'price_momentum': np.random.uniform(-0.05, 0.05, days),
            'volatility': np.random.uniform(0.15, 0.35, days),
            'delta_change': np.random.uniform(-0.1, 0.1, days),
            'gamma_exposure': np.random.uniform(0, 0.05, days),
            'iv_percentile': np.random.uniform(10, 90, days)
        }

        return pd.DataFrame(data)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML training"""
        try:
            # Ensure all required columns exist
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0

            # Create additional features
            df['price_change'] = df['close'].pct_change()
            df['volume_ma'] = df['volume'].rolling(window=5).mean()
            df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=10).mean()

            # Create target variable (1: Buy, 0: Hold, -1: Sell)
            df['target'] = 0
            df.loc[(df['price_change'] > 0.01) & (df['volume_ratio'] > 1.2), 'target'] = 1
            df.loc[(df['price_change'] < -0.01) & (df['volume_ratio'] > 1.2), 'target'] = -1

            # Forward fill missing values
            df = df.fillna(method='ffill').fillna(0)

            return df

        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            raise

    def train_models(self, symbol: str = 'NIFTY') -> Dict:
        """Train ML models with market data"""
        try:
            self.logger.info(f"Starting model training for {symbol}")

            # Fetch market data
            df = self.fetch_market_data(symbol)
            if df.empty:
                raise ValueError("No market data available for training")

            # Prepare features
            df = self.prepare_features(df)

            # Prepare training data
            X = df[self.feature_columns].values
            y = df['target'].values

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y,
                test_size=self.config['training_params']['test_size'],
                random_state=42,
                stratify=y
            )

            # Train individual models
            results = {}
            for name, model in self.models.items():
                self.logger.info(f"Training {name} model")
                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {
                    'accuracy': accuracy,
                    'model': model
                }

                self.logger.info(f"{name} accuracy: {accuracy:.3f}")

            # Train ensemble model
            self.logger.info("Training ensemble model")
            self.ensemble_model.fit(X_train, y_train)

            # Evaluate ensemble
            y_pred_ensemble = self.ensemble_model.predict(X_test)
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            results['ensemble'] = {
                'accuracy': ensemble_accuracy,
                'model': self.ensemble_model
            }

            self.logger.info(f"Ensemble accuracy: {ensemble_accuracy:.3f}")

            # Save models
            self._save_models()

            self.is_trained = True
            self.logger.info("Model training completed successfully")

            return results

        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            raise

    def _save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs('ml_bot/models', exist_ok=True)

            # Save individual models
            for name, model in self.models.items():
                joblib.dump(model, f'ml_bot/models/{name}_model.pkl')

            # Save ensemble model
            joblib.dump(self.ensemble_model, 'ml_bot/models/ensemble_model.pkl')

            # Save scaler
            joblib.dump(self.scaler, 'ml_bot/models/scaler.pkl')

            self.logger.info("Models saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving models: {e}")

    def load_models(self):
        """Load trained models from disk"""
        try:
            # Load individual models
            for name in self.models.keys():
                model_path = f'ml_bot/models/{name}_model.pkl'
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)

            # Load ensemble model
            ensemble_path = 'ml_bot/models/ensemble_model.pkl'
            if os.path.exists(ensemble_path):
                self.ensemble_model = joblib.load(ensemble_path)

            # Load scaler
            scaler_path = 'ml_bot/models/scaler.pkl'
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

            self.is_trained = True
            self.logger.info("Models loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

    def generate_prediction(self, current_data: Dict) -> MLSignal:
        """Generate ML prediction for current market data"""
        try:
            if not self.is_trained:
                raise ValueError("Models not trained yet")

            # Prepare features
            features = []
            for col in self.feature_columns:
                features.append(current_data.get(col, 0))

            # Scale features
            features_scaled = self.scaler.transform([features])

            # Get ensemble prediction
            prediction = self.ensemble_model.predict(features_scaled)[0]
            probabilities = self.ensemble_model.predict_proba(features_scaled)[0]

            # Get confidence (max probability)
            confidence = np.max(probabilities)

            # Determine action
            if prediction == 1 and confidence > self.config['training_params']['min_confidence']:
                action = 'BUY_CE'
                reasoning = f"Bullish signal with {confidence:.1%} confidence"
            elif prediction == -1 and confidence > self.config['training_params']['min_confidence']:
                action = 'BUY_PE'
                reasoning = f"Bearish signal with {confidence:.1%} confidence"
            else:
                action = 'WAIT'
                reasoning = f"Low confidence signal ({confidence:.1%})"

            # Create ML signal
            signal = MLSignal(
                symbol=current_data.get('symbol', 'NIFTY'),
                action=action,
                confidence=confidence,
                prediction_source='ML Ensemble',
                features_used=self.feature_columns,
                timestamp=datetime.now().isoformat(),
                reasoning=reasoning,
                target_strike=current_data.get('current_price', 0),
                expected_move=abs(prediction) * 0.02  # 2% expected move
            )

            return signal

        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            raise

    async def send_signal_to_main_system(self, signal: MLSignal):
        """Send ML signal to main trading system"""
        try:
            signal_data = {
                'type': 'ml_signal',
                'data': {
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'confidence': signal.confidence,
                    'source': signal.prediction_source,
                    'timestamp': signal.timestamp,
                    'reasoning': signal.reasoning,
                    'target_strike': signal.target_strike,
                    'expected_move': signal.expected_move
                }
            }

            # Send via WebSocket
            async with websockets.connect(self.websocket_url) as websocket:
                await websocket.send(json.dumps(signal_data))
                response = await websocket.recv()
                self.logger.info(f"Signal sent successfully: {response}")

        except Exception as e:
            # Fallback to HTTP API
            try:
                response = requests.post(
                    f"{self.main_system_api}/api/ml_signal",
                    json=signal_data,
                    timeout=10
                )
                if response.status_code == 200:
                    self.logger.info("Signal sent via HTTP API")
                else:
                    self.logger.error(f"HTTP API error: {response.status_code}")
            except Exception as http_error:
                self.logger.error(f"Failed to send signal: {e}, HTTP fallback: {http_error}")

    async def run_continuous_analysis(self):
        """Run continuous market analysis and signal generation"""
        self.logger.info("Starting continuous ML analysis")

        while True:
            try:
                # Fetch current market data
                current_data = self._get_current_market_data()

                # Generate prediction
                signal = self.generate_prediction(current_data)

                # Send signal if actionable
                if signal.action != 'WAIT':
                    await self.send_signal_to_main_system(signal)
                    self.logger.info(f"Generated signal: {signal.action} with {signal.confidence:.1%} confidence")

                # Wait before next analysis
                await asyncio.sleep(30)  # Analyze every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in continuous analysis: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    def _get_current_market_data(self) -> Dict:
        """Get current market data for analysis"""
        try:
            # This would fetch real-time data from main system
            # For now, return sample data
            return {
                'symbol': 'NIFTY',
                'current_price': 23500,
                'rsi': 62,
                'ema_short': 23520,
                'ema_long': 23480,
                'vwap_ratio': 1.01,
                'volume_ratio': 1.5,
                'price_momentum': 0.015,
                'volatility': 0.22,
                'delta_change': 0.02,
                'gamma_exposure': 0.03,
                'iv_percentile': 45
            }

        except Exception as e:
            self.logger.error(f"Error getting current market data: {e}")
            return {}

def main():
    """Main function to run ML bot"""
    try:
        # Initialize ML bot
        bot = MLTradingBot()
        bot.initialize_models()

        # Try to load existing models
        try:
            bot.load_models()
            print("✅ Existing models loaded successfully")
        except:
            print("📚 Training new models...")
            bot.train_models()
            print("✅ Models trained successfully")

        # Run continuous analysis
        print("🤖 Starting ML Bot...")
        asyncio.run(bot.run_continuous_analysis())

    except KeyboardInterrupt:
        print("\n🛑 ML Bot stopped by user")
    except Exception as e:
        print(f"❌ ML Bot error: {e}")

if __name__ == "__main__":
    main()