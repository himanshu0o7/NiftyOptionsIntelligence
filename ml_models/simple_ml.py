import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
from utils.logger import Logger

class SimplifiedMLEngine:
    """Simplified ML engine that works without OpenMP dependencies"""
    
    def __init__(self):
        self.logger = Logger()
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_performance = {}
        self.feature_names = []
        
        # Initialize basic models that work without OpenMP
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models that don't require OpenMP"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=20,  # Small number for performance
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=1  # Single thread to avoid OpenMP
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='lbfgs',  # Better solver for multiclass
                multi_class='ovr'  # One-vs-Rest for multiclass
            ),
            'svm': SVC(
                kernel='rbf',  # RBF kernel for better performance
                probability=True,
                random_state=42,
                max_iter=1000,
                gamma='scale'  # Better gamma setting
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,  # Increased iterations
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create simple but effective trading features"""
        try:
            features_df = data.copy()
            
            # Price features
            features_df['price_change'] = features_df['close'].pct_change()
            features_df['price_momentum'] = features_df['close'].pct_change(periods=5)
            features_df['volatility'] = features_df['price_change'].rolling(10).std()
            
            # Volume features
            features_df['volume_avg'] = features_df['volume'].rolling(10).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_avg']
            
            # Simple moving averages
            features_df['sma_5'] = features_df['close'].rolling(5).mean()
            features_df['sma_20'] = features_df['close'].rolling(20).mean()
            features_df['sma_ratio'] = features_df['sma_5'] / features_df['sma_20']
            
            # Price position
            features_df['high_low_ratio'] = features_df['high'] / features_df['low']
            features_df['close_high_ratio'] = features_df['close'] / features_df['high']
            
            # RSI-like indicator
            delta = features_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features_df['rsi_simple'] = 100 - (100 / (1 + rs))
            
            # Select feature columns
            self.feature_names = [
                'price_change', 'price_momentum', 'volatility',
                'volume_ratio', 'sma_ratio', 'high_low_ratio',
                'close_high_ratio', 'rsi_simple'
            ]
            
            # Fill NaN values
            for col in self.feature_names:
                features_df[col] = features_df[col].fillna(features_df[col].median())
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Feature creation error: {str(e)}")
            return data
    
    def prepare_training_data(self, data: pd.DataFrame, lookback: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with labels"""
        try:
            # Create features
            features_df = self.create_features(data)
            
            # Create labels based on future price movement
            features_df['future_return'] = features_df['close'].shift(-lookback) / features_df['close'] - 1
            
            # Define labels: 0=SELL, 1=HOLD, 2=BUY
            def create_labels(returns, buy_thresh=0.01, sell_thresh=-0.01):
                labels = []
                for ret in returns:
                    if pd.isna(ret):
                        labels.append(1)  # HOLD
                    elif ret > buy_thresh:
                        labels.append(2)  # BUY
                    elif ret < sell_thresh:
                        labels.append(0)  # SELL
                    else:
                        labels.append(1)  # HOLD
                return labels
            
            features_df['label'] = create_labels(features_df['future_return'])
            
            # Remove NaN rows
            features_df = features_df[:-lookback].dropna()
            
            X = features_df[self.feature_names]
            y = features_df['label']
            
            self.logger.info(f"Training data prepared: {len(X)} samples, {len(self.feature_names)} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Training data preparation error: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def train_models(self, data: pd.DataFrame) -> Dict:
        """Train all ML models"""
        try:
            # Prepare data
            X, y = self.prepare_training_data(data)
            
            if len(X) == 0:
                return {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            results = {}
            
            # Train each model
            for name, model in self.models.items():
                try:
                    self.logger.info(f"Training {name}...")
                    
                    # Use scaled features for models that benefit from scaling
                    if name in ['logistic_regression', 'svm', 'neural_network']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Calculate metrics with proper error handling
                    try:
                        from sklearn.metrics import precision_recall_fscore_support
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
                    except Exception as metric_error:
                        self.logger.warning(f"Metric calculation error for {name}: {metric_error}")
                        results[name] = {
                            'accuracy': accuracy,
                            'trained': True
                        }
                    
                    self.logger.info(f"{name} accuracy: {accuracy:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {name}: {str(e)}")
                    continue
            
            self.model_performance = results
            self.is_trained = True
            
            # Find best model
            if results:
                self.best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
                self.logger.info(f"Best model: {self.best_model}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Model training error: {str(e)}")
            return {}
    
    def generate_signals(self, current_data: pd.DataFrame, min_confidence: float = 0.6) -> List[Dict]:
        """Generate trading signals"""
        try:
            if not self.is_trained:
                self.logger.warning("Models not trained yet")
                return []
            
            # Create features for current data
            features_df = self.create_features(current_data)
            
            # Get latest features
            latest_features = features_df[self.feature_names].iloc[-1:].fillna(0)
            
            # Scale features
            latest_scaled = pd.DataFrame(
                self.scaler.transform(latest_features),
                columns=latest_features.columns
            )
            
            signals = []
            predictions = {}
            
            # Get predictions from each model
            for name, model in self.models.items():
                try:
                    if name in ['logistic_regression', 'svm', 'neural_network']:
                        pred = model.predict(latest_scaled)[0]
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(latest_scaled)[0]
                            confidence = np.max(proba)
                        else:
                            confidence = 0.7
                    else:
                        pred = model.predict(latest_features)[0]
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(latest_features)[0]
                            confidence = np.max(proba)
                        else:
                            confidence = 0.7
                    
                    predictions[name] = {'prediction': pred, 'confidence': confidence}
                    
                except Exception as e:
                    self.logger.error(f"Prediction error for {name}: {str(e)}")
                    continue
            
            # Ensemble prediction (majority voting)
            if predictions:
                pred_values = [p['prediction'] for p in predictions.values()]
                confidences = [p['confidence'] for p in predictions.values()]
                
                ensemble_pred = max(set(pred_values), key=pred_values.count)
                ensemble_confidence = np.mean(confidences)
                
                # Create signal if confidence is high enough
                if ensemble_confidence >= min_confidence and ensemble_pred != 1:  # Not HOLD
                    action = 'BUY' if ensemble_pred == 2 else 'SELL'
                    
                    signal = {
                        'timestamp': datetime.now(),
                        'symbol': 'NIFTY',
                        'action': action,
                        'confidence': ensemble_confidence,
                        'signal_type': 'SIMPLIFIED_ML',
                        'model_predictions': predictions,
                        'source': 'Simplified ML Engine'
                    }
                    
                    signals.append(signal)
                    self.logger.info(f"Generated signal: {action} with confidence {ensemble_confidence:.3f}")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {str(e)}")
            return []
    
    def get_model_status(self) -> Dict:
        """Get model status"""
        return {
            'is_trained': self.is_trained,
            'num_models': len(self.models),
            'feature_count': len(self.feature_names),
            'model_performance': self.model_performance,
            'best_model': getattr(self, 'best_model', None)
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        if not self.model_performance:
            return {}
        
        metrics = {}
        for model_name, perf in self.model_performance.items():
            metrics[model_name] = {
                'accuracy': perf.get('accuracy', 0),
                'total_signals': 100,  # Placeholder
                'correct_signals': int(perf.get('accuracy', 0) * 100)
            }
        
        return metrics
    
    def save_models(self, filepath: str = 'ml_models/simple_models'):
        """Save models"""
        try:
            os.makedirs(filepath, exist_ok=True)
            
            for name, model in self.models.items():
                joblib.dump(model, f"{filepath}/{name}.joblib")
            
            joblib.dump(self.scaler, f"{filepath}/scaler.joblib")
            
            metadata = {
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'model_performance': self.model_performance
            }
            joblib.dump(metadata, f"{filepath}/metadata.joblib")
            
            self.logger.info("Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Model saving error: {str(e)}")
    
    def load_models(self, filepath: str = 'ml_models/simple_models'):
        """Load models"""
        try:
            if os.path.exists(f"{filepath}/metadata.joblib"):
                metadata = joblib.load(f"{filepath}/metadata.joblib")
                self.is_trained = metadata['is_trained']
                self.feature_names = metadata['feature_names']
                self.model_performance = metadata['model_performance']
            
            if os.path.exists(f"{filepath}/scaler.joblib"):
                self.scaler = joblib.load(f"{filepath}/scaler.joblib")
            
            for name in self.models.keys():
                if os.path.exists(f"{filepath}/{name}.joblib"):
                    self.models[name] = joblib.load(f"{filepath}/{name}.joblib")
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Model loading error: {str(e)}")