"""
Improved Ensemble Predictor for Options Trading
Addresses ML model warnings and performance issues
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import warnings
import joblib
import os
from datetime import datetime
from utils.logger import Logger

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class EnsemblePredictor:
    """Improved ensemble predictor with better error handling and performance"""

    def __init__(self):
        self.logger = Logger()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.models = {}
        self.ensemble_model = None
        self.performance_metrics = {}
        self.feature_names = []

        # Initialize improved models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize optimized ML models"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1,
                class_weight='balanced'  # Handle class imbalance
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='lbfgs',
                multi_class='ovr',
                class_weight='balanced'
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                gamma='scale',
                class_weight='balanced'
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                alpha=0.01  # Regularization
            )
        }

    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive trading features"""
        try:
            features_df = data.copy()

            # Price-based features
            features_df['returns'] = features_df['close'].pct_change()
            features_df['log_returns'] = np.log(features_df['close']).diff()
            features_df['price_momentum_5'] = features_df['close'].pct_change(5)
            features_df['price_momentum_10'] = features_df['close'].pct_change(10)

            # Volatility features
            features_df['volatility_5'] = features_df['returns'].rolling(5).std()
            features_df['volatility_10'] = features_df['returns'].rolling(10).std()
            features_df['volatility_20'] = features_df['returns'].rolling(20).std()

            # Volume features
            features_df['volume_sma_5'] = features_df['volume'].rolling(5).mean()
            features_df['volume_sma_10'] = features_df['volume'].rolling(10).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma_10']

            # Technical indicators
            features_df['sma_5'] = features_df['close'].rolling(5).mean()
            features_df['sma_10'] = features_df['close'].rolling(10).mean()
            features_df['sma_20'] = features_df['close'].rolling(20).mean()

            # Price relative to moving averages
            features_df['price_vs_sma5'] = features_df['close'] / features_df['sma_5']
            features_df['price_vs_sma10'] = features_df['close'] / features_df['sma_10']
            features_df['price_vs_sma20'] = features_df['close'] / features_df['sma_20']

            # Bollinger Bands
            features_df['bb_middle'] = features_df['close'].rolling(20).mean()
            features_df['bb_std'] = features_df['close'].rolling(20).std()
            features_df['bb_upper'] = features_df['bb_middle'] + (features_df['bb_std'] * 2)
            features_df['bb_lower'] = features_df['bb_middle'] - (features_df['bb_std'] * 2)
            features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])

            # RSI-like momentum
            delta = features_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features_df['rsi'] = 100 - (100 / (1 + rs))

            # Select final features
            self.feature_names = [
                'returns', 'price_momentum_5', 'price_momentum_10',
                'volatility_5', 'volatility_10', 'volume_ratio',
                'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20',
                'bb_position', 'rsi'
            ]

            return features_df

        except Exception as e:
            self.logger.error(f"Feature creation error: {e}")
            return pd.DataFrame()

    def prepare_training_data(self, data: pd.DataFrame, lookback: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with improved labeling"""
        try:
            # Create features
            features_df = self.create_advanced_features(data)

            if features_df.empty:
                return pd.DataFrame(), pd.Series()

            # Create future returns for labeling
            features_df['future_return'] = features_df['close'].pct_change(lookback).shift(-lookback)

            # Improved labeling logic
            def create_improved_labels(returns: pd.Series) -> List[int]:
                labels = []
                # Use dynamic thresholds based on volatility
                volatility = returns.rolling(20).std()

                for i, ret in enumerate(returns):
                    if pd.isna(ret) or pd.isna(volatility.iloc[i]):
                        labels.append(1)  # HOLD
                    else:
                        vol_threshold = volatility.iloc[i] * 0.5
                        if ret > vol_threshold:
                            labels.append(2)  # BUY
                        elif ret < -vol_threshold:
                            labels.append(0)  # SELL
                        else:
                            labels.append(1)  # HOLD
                return labels

            features_df['label'] = create_improved_labels(features_df['future_return'])

            # Remove rows with NaN values
            features_df = features_df[:-lookback].dropna()

            # Extract features and labels
            X = features_df[self.feature_names]
            y = features_df['label']

            # Ensure we have enough samples and class diversity
            if len(X) < 50:
                self.logger.warning("Insufficient training data")
                return pd.DataFrame(), pd.Series()

            unique_classes = len(np.unique(y))
            if unique_classes < 2:
                self.logger.warning("Insufficient class diversity")
                return pd.DataFrame(), pd.Series()

            self.logger.info(f"Training data prepared: {len(X)} samples, {len(self.feature_names)} features, {unique_classes} classes")
            return X, y

        except Exception as e:
            self.logger.error(f"Training data preparation error: {e}")
            return pd.DataFrame(), pd.Series()

    def train_models(self, data: pd.DataFrame) -> Dict:
        """Train ensemble models with improved error handling"""
        try:
            # Prepare training data
            X, y = self.prepare_training_data(data)

            if len(X) == 0:
                return {"error": "No training data available"}

            # Split data with stratification
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError:
                # If stratification fails, use random split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Convert back to DataFrame for easier handling
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names)

            results = {}
            trained_models = []

            # Train individual models
            for name, model in self.models.items():
                try:
                    self.logger.info(f"Training {name}...")

                    # Check if we have enough samples for each class
                    class_counts = np.bincount(y_train)
                    if len(class_counts) < 2 or min(class_counts) < 2:
                        self.logger.warning(f"Insufficient samples per class for {name}")
                        continue

                    # Use scaled features for models that benefit from scaling
                    if name in ['logistic_regression', 'svm', 'neural_network']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)

                    # Calculate precision, recall, f1 with proper error handling
                    try:
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            y_test, y_pred, average='weighted', zero_division=0
                        )

                        results[name] = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'trained': True
                        }

                        # Add to trained models for ensemble
                        trained_models.append((name, model))

                    except Exception as metric_error:
                        self.logger.warning(f"Metric calculation error for {name}: {metric_error}")
                        results[name] = {
                            'accuracy': accuracy,
                            'trained': True
                        }

                    self.logger.info(f"{name} accuracy: {accuracy:.4f}")

                except Exception as model_error:
                    self.logger.error(f"Error training {name}: {model_error}")
                    results[name] = {
                        'accuracy': 0,
                        'error': str(model_error),
                        'trained': False
                    }
                    continue

            # Create ensemble if we have trained models
            if len(trained_models) >= 2:
                try:
                    # Create voting classifier
                    voting_models = [(name, model) for name, model in trained_models]
                    self.ensemble_model = VotingClassifier(
                        estimators=voting_models,
                        voting='soft'  # Use probability voting
                    )

                    # Train ensemble
                    self.ensemble_model.fit(X_train_scaled, y_train)

                    # Test ensemble
                    ensemble_pred = self.ensemble_model.predict(X_test_scaled)
                    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

                    results['ensemble'] = {
                        'accuracy': ensemble_accuracy,
                        'trained': True,
                        'models_used': len(trained_models)
                    }

                    self.logger.info(f"Ensemble accuracy: {ensemble_accuracy:.4f}")

                except Exception as ensemble_error:
                    self.logger.error(f"Ensemble creation error: {ensemble_error}")

            self.performance_metrics = results
            self.is_trained = True

            # Find best individual model
            trained_results = {k: v for k, v in results.items() if v.get('trained', False)}
            if trained_results:
                self.best_model = max(trained_results.keys(), key=lambda k: trained_results[k]['accuracy'])
                self.logger.info(f"Best model: {self.best_model}")

            return results

        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            return {"error": str(e)}

    def generate_ensemble_signals(self, current_data: pd.DataFrame, min_confidence: float = 0.65) -> List[Dict]:
        """Generate signals using ensemble approach"""
        try:
            if not self.is_trained:
                self.logger.warning("Models not trained yet")
                return []

            # Create features
            features_df = self.create_advanced_features(current_data)

            if features_df.empty:
                return []

            # Get latest features
            latest_features = features_df[self.feature_names].iloc[-1:].fillna(0)

            # Scale features
            latest_features_scaled = self.scaler.transform(latest_features)

            signals = []

            # Generate ensemble prediction if available
            if self.ensemble_model is not None:
                try:
                    prediction = self.ensemble_model.predict(latest_features_scaled)[0]
                    probabilities = self.ensemble_model.predict_proba(latest_features_scaled)[0]
                    confidence = max(probabilities)

                    if confidence >= min_confidence:
                        action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

                        signals.append({
                            'signal_type': 'ensemble_ml',
                            'action': action_map[prediction],
                            'confidence': confidence,
                            'timestamp': datetime.now(),
                            'method': 'ensemble_voting',
                            'features_used': self.feature_names
                        })

                except Exception as e:
                    self.logger.error(f"Ensemble prediction error: {e}")

            # Generate individual model signals as backup
            for name, model in self.models.items():
                try:
                    if name in ['logistic_regression', 'svm', 'neural_network']:
                        prediction = model.predict(latest_features_scaled)[0]
                        probabilities = model.predict_proba(latest_features_scaled)[0]
                    else:
                        prediction = model.predict(latest_features)[0]
                        probabilities = model.predict_proba(latest_features)[0]

                    confidence = max(probabilities)

                    if confidence >= min_confidence:
                        action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

                        signals.append({
                            'signal_type': f'ml_{name}',
                            'action': action_map[prediction],
                            'confidence': confidence,
                            'timestamp': datetime.now(),
                            'method': name,
                            'features_used': self.feature_names
                        })

                except Exception as e:
                    self.logger.error(f"Signal generation error for {name}: {e}")
                    continue

            return signals

        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return []

    def get_model_performance(self) -> Dict:
        """Get model performance metrics"""
        return self.performance_metrics

    def save_models(self, filepath: str = "ml_models/ensemble_models.pkl"):
        """Save trained models"""
        try:
            model_data = {
                'models': self.models,
                'ensemble_model': self.ensemble_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'performance_metrics': self.performance_metrics,
                'is_trained': self.is_trained
            }

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(model_data, filepath)
            self.logger.info(f"Models saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Model saving error: {e}")
            return False

    def load_models(self, filepath: str = "ml_models/ensemble_models.pkl"):
        """Load trained models"""
        try:
            if not os.path.exists(filepath):
                self.logger.warning(f"Model file not found: {filepath}")
                return False

            model_data = joblib.load(filepath)

            self.models = model_data['models']
            self.ensemble_model = model_data['ensemble_model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.performance_metrics = model_data['performance_metrics']
            self.is_trained = model_data['is_trained']

            self.logger.info(f"Models loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
            return False