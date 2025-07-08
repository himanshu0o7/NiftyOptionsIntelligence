import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from datetime import datetime
from utils.logger import Logger

class LSTMSignalGenerator:
    """LSTM neural network for time series prediction and signal generation"""
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 5):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        self.logger = Logger()
        self.feature_columns = []
        self.training_history = None
    
    def _create_lstm_model(self, n_features: int, n_outputs: int = 1) -> keras.Model:
        """Create LSTM model architecture"""
        model = keras.Sequential([
            # First LSTM layer with return sequences
            layers.LSTM(
                units=100,
                return_sequences=True,
                input_shape=(self.sequence_length, n_features),
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(
                units=80,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            layers.BatchNormalization(),
            
            # Third LSTM layer
            layers.LSTM(
                units=60,
                return_sequences=False,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(units=50, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(units=25, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(units=n_outputs)
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_bidirectional_lstm(self, n_features: int, n_outputs: int = 1) -> keras.Model:
        """Create bidirectional LSTM model for better pattern recognition"""
        model = keras.Sequential([
            # Bidirectional LSTM layers
            layers.Bidirectional(
                layers.LSTM(
                    units=100,
                    return_sequences=True,
                    dropout=0.2,
                    recurrent_dropout=0.2
                ),
                input_shape=(self.sequence_length, n_features)
            ),
            layers.BatchNormalization(),
            
            layers.Bidirectional(
                layers.LSTM(
                    units=80,
                    return_sequences=True,
                    dropout=0.2,
                    recurrent_dropout=0.2
                )
            ),
            layers.BatchNormalization(),
            
            layers.Bidirectional(
                layers.LSTM(
                    units=60,
                    dropout=0.2,
                    recurrent_dropout=0.2
                )
            ),
            layers.BatchNormalization(),
            
            # Attention mechanism (simplified)
            layers.Dense(units=120, activation='tanh'),
            layers.Dropout(0.3),
            
            # Output layers
            layers.Dense(units=50, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(units=n_outputs)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _prepare_lstm_data(self, data: pd.DataFrame, 
                          target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        try:
            # Select features for training
            if not self.feature_columns:
                self.feature_columns = [
                    'open', 'high', 'low', 'close', 'volume',
                    'rsi', 'macd', 'bb_position', 'volatility'
                ]
                # Filter to only existing columns
                self.feature_columns = [col for col in self.feature_columns if col in data.columns]
            
            # Prepare feature data
            feature_data = data[self.feature_columns].values
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(feature_data)
            
            # Create sequences
            X, y = [], []
            
            for i in range(self.sequence_length, len(scaled_data) - self.prediction_horizon):
                # Input sequence
                X.append(scaled_data[i-self.sequence_length:i])
                
                # Target (future price movement)
                current_price = data[target_column].iloc[i]
                future_price = data[target_column].iloc[i + self.prediction_horizon]
                price_change = (future_price - current_price) / current_price
                y.append(price_change)
            
            X = np.array(X)
            y = np.array(y)
            
            self.logger.info(f"LSTM data prepared: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"LSTM data preparation error: {str(e)}")
            return np.array([]), np.array([])
    
    def train_lstm(self, data: pd.DataFrame, model_type: str = 'bidirectional',
                   validation_split: float = 0.2, epochs: int = 100) -> Dict:
        """Train LSTM model"""
        try:
            if not TF_AVAILABLE:
                self.logger.warning("TensorFlow not available, cannot train LSTM")
                return {'error': 'TensorFlow not available'}
            
            # Prepare data
            X, y = self._prepare_lstm_data(data)
            
            if len(X) == 0:
                self.logger.error("No data available for LSTM training")
                return {}
            
            # Create model
            n_features = X.shape[2]
            
            if model_type == 'bidirectional':
                self.model = self._create_bidirectional_lstm(n_features)
            else:
                self.model = self._create_lstm_model(n_features)
            
            # Define callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=0.0001
            )
            
            # Train model
            self.logger.info(f"Training LSTM model with {len(X)} samples...")
            
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=32,
                validation_split=validation_split,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            self.training_history = history.history
            self.is_trained = True
            
            # Calculate final metrics
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            results = {
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'epochs_trained': len(history.history['loss']),
                'model_type': model_type,
                'n_features': n_features,
                'sequence_length': self.sequence_length
            }
            
            self.logger.info(f"LSTM training completed. Val loss: {final_val_loss:.6f}")
            return results
            
        except Exception as e:
            self.logger.error(f"LSTM training error: {str(e)}")
            return {}
    
    def predict_price_movement(self, recent_data: pd.DataFrame) -> Dict:
        """Predict future price movement using trained LSTM"""
        try:
            if not self.is_trained or self.model is None:
                self.logger.warning("LSTM model not trained")
                return {}
            
            # Prepare recent data
            if len(recent_data) < self.sequence_length:
                self.logger.warning(f"Need at least {self.sequence_length} data points for prediction")
                return {}
            
            # Get last sequence
            feature_data = recent_data[self.feature_columns].tail(self.sequence_length).values
            scaled_data = self.scaler.transform(feature_data)
            
            # Reshape for LSTM
            X_pred = scaled_data.reshape(1, self.sequence_length, len(self.feature_columns))
            
            # Make prediction
            predicted_change = self.model.predict(X_pred, verbose=0)[0][0]
            
            # Calculate confidence based on recent model performance
            confidence = min(max(abs(predicted_change) * 10, 0.1), 0.9)
            
            prediction_result = {
                'predicted_change': float(predicted_change),
                'confidence': float(confidence),
                'prediction_horizon': self.prediction_horizon,
                'timestamp': datetime.now()
            }
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"LSTM prediction error: {str(e)}")
            return {}
    
    def generate_lstm_signals(self, recent_data: pd.DataFrame,
                             change_threshold: float = 0.015) -> List[Dict]:
        """Generate trading signals based on LSTM predictions"""
        try:
            prediction = self.predict_price_movement(recent_data)
            
            if not prediction:
                return []
            
            signals = []
            predicted_change = prediction['predicted_change']
            confidence = prediction['confidence']
            
            # Generate signal based on predicted change
            if abs(predicted_change) > change_threshold and confidence > 0.6:
                if predicted_change > 0:
                    action = 'BUY'
                else:
                    action = 'SELL'
                
                signal = {
                    'timestamp': datetime.now(),
                    'symbol': 'NIFTY',  # Would be dynamic
                    'action': action,
                    'confidence': confidence,
                    'signal_type': 'LSTM_PREDICTION',
                    'predicted_change': predicted_change,
                    'prediction_horizon': self.prediction_horizon,
                    'source': 'LSTM Signal Generator'
                }
                
                signals.append(signal)
                self.logger.info(f"LSTM signal: {action} (change: {predicted_change:.4f}, confidence: {confidence:.3f})")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"LSTM signal generation error: {str(e)}")
            return []
    
    def create_ensemble_prediction(self, recent_data: pd.DataFrame,
                                 models: List[keras.Model]) -> Dict:
        """Create ensemble prediction using multiple LSTM models"""
        try:
            if not models:
                return self.predict_price_movement(recent_data)
            
            predictions = []
            confidences = []
            
            # Get prediction from each model
            for model in models:
                if len(recent_data) < self.sequence_length:
                    continue
                
                feature_data = recent_data[self.feature_columns].tail(self.sequence_length).values
                scaled_data = self.scaler.transform(feature_data)
                X_pred = scaled_data.reshape(1, self.sequence_length, len(self.feature_columns))
                
                pred = model.predict(X_pred, verbose=0)[0][0]
                predictions.append(pred)
                confidences.append(min(max(abs(pred) * 10, 0.1), 0.9))
            
            if not predictions:
                return {}
            
            # Weighted average of predictions
            weights = np.array(confidences)
            weights = weights / weights.sum()
            
            ensemble_prediction = np.average(predictions, weights=weights)
            ensemble_confidence = np.mean(confidences)
            
            return {
                'predicted_change': float(ensemble_prediction),
                'confidence': float(ensemble_confidence),
                'individual_predictions': predictions,
                'individual_confidences': confidences,
                'n_models': len(models),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction error: {str(e)}")
            return {}
    
    def save_lstm_model(self, filepath: str = 'ml_models/lstm_models'):
        """Save LSTM model and scaler"""
        try:
            os.makedirs(filepath, exist_ok=True)
            
            if self.model is not None:
                self.model.save(f"{filepath}/lstm_model.h5")
            
            # Save scaler and metadata
            joblib.dump(self.scaler, f"{filepath}/lstm_scaler.joblib")
            
            metadata = {
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'is_trained': self.is_trained,
                'feature_columns': self.feature_columns,
                'training_history': self.training_history
            }
            joblib.dump(metadata, f"{filepath}/lstm_metadata.joblib")
            
            self.logger.info(f"LSTM model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"LSTM model saving error: {str(e)}")
    
    def load_lstm_model(self, filepath: str = 'ml_models/lstm_models'):
        """Load LSTM model and scaler"""
        try:
            # Load model
            if os.path.exists(f"{filepath}/lstm_model.h5"):
                self.model = keras.models.load_model(f"{filepath}/lstm_model.h5")
            
            # Load scaler
            if os.path.exists(f"{filepath}/lstm_scaler.joblib"):
                self.scaler = joblib.load(f"{filepath}/lstm_scaler.joblib")
            
            # Load metadata
            if os.path.exists(f"{filepath}/lstm_metadata.joblib"):
                metadata = joblib.load(f"{filepath}/lstm_metadata.joblib")
                self.sequence_length = metadata['sequence_length']
                self.prediction_horizon = metadata['prediction_horizon']
                self.is_trained = metadata['is_trained']
                self.feature_columns = metadata['feature_columns']
                self.training_history = metadata.get('training_history', None)
            
            self.logger.info(f"LSTM model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"LSTM model loading error: {str(e)}")
    
    def get_training_plots_data(self) -> Dict:
        """Get data for plotting training history"""
        try:
            if not self.training_history:
                return {}
            
            return {
                'loss': self.training_history.get('loss', []),
                'val_loss': self.training_history.get('val_loss', []),
                'mae': self.training_history.get('mae', []),
                'val_mae': self.training_history.get('val_mae', [])
            }
            
        except Exception as e:
            self.logger.error(f"Training plots data error: {str(e)}")
            return {}
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate LSTM model performance"""
        try:
            if not self.is_trained or self.model is None:
                return {}
            
            # Prepare test data
            X_test, y_test = self._prepare_lstm_data(test_data)
            
            if len(X_test) == 0:
                return {}
            
            # Make predictions
            y_pred = self.model.predict(X_test, verbose=0)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Direction accuracy (buy/sell/hold)
            direction_actual = np.sign(y_test)
            direction_pred = np.sign(y_pred.flatten())
            direction_accuracy = np.mean(direction_actual == direction_pred)
            
            evaluation = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'direction_accuracy': float(direction_accuracy),
                'n_samples': len(y_test)
            }
            
            self.logger.info(f"LSTM evaluation - RMSE: {rmse:.6f}, Direction Accuracy: {direction_accuracy:.3f}")
            return evaluation
            
        except Exception as e:
            self.logger.error(f"LSTM evaluation error: {str(e)}")
            return {}