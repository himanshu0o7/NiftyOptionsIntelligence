import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import asyncio
import threading
import time
from utils.logger import Logger
from .signal_models import MLSignalGenerator
from .lstm_model import LSTMSignalGenerator
from .feature_engineering import FeatureEngineer

class MLTradingEngine:
    """Main ML trading engine that integrates all ML models"""

    def __init__(self, retrain_frequency: int = 7):  # Retrain weekly
        self.ml_signal_generator = MLSignalGenerator()
        self.lstm_generator = LSTMSignalGenerator()
        self.feature_engineer = FeatureEngineer()
        self.logger = Logger()

        self.retrain_frequency = retrain_frequency  # days
        self.last_training = None
        self.is_running = False
        self.training_thread = None

        # Performance tracking
        self.signal_performance = {
            'ml_ensemble': {'correct': 0, 'total': 0},
            'lstm': {'correct': 0, 'total': 0},
            'combined': {'correct': 0, 'total': 0}
        }

    def initialize_models(self, historical_data: pd.DataFrame,
                         options_data: pd.DataFrame = None,
                         initial_training: bool = True) -> Dict:
        """Initialize and train all ML models"""
        try:
            results = {}

            if initial_training:
                self.logger.info("Starting initial ML model training...")

                # Train ML ensemble models
                self.logger.info("Training ML ensemble models...")
                X, y = self.ml_signal_generator.prepare_training_data(historical_data, options_data)

                if len(X) > 0:
                    ml_results = self.ml_signal_generator.train_models(X, y)
                    results['ml_ensemble'] = ml_results

                    # Hyperparameter optimization for best model
                    self.logger.info("Optimizing hyperparameters...")
                    opt_results = self.ml_signal_generator.hyperparameter_optimization(X, y, 'xgboost')
                    results['hyperparameter_optimization'] = opt_results

                # Train LSTM model
                self.logger.info("Training LSTM model...")
                lstm_results = self.lstm_generator.train_lstm(historical_data, model_type='bidirectional')
                results['lstm'] = lstm_results

                # Save models
                self.save_all_models()

                self.last_training = datetime.now()
                self.logger.info("Initial ML model training completed")

            else:
                # Load existing models
                self.load_all_models()
                self.logger.info("Loaded existing ML models")

            return results

        except Exception as e:
            self.logger.error(f"ML model initialization error: {str(e)}")
            return {}

    def generate_ml_signals(self, current_data: pd.DataFrame,
                           options_data: pd.DataFrame = None,
                           min_confidence: float = 0.65) -> List[Dict]:
        """Generate signals from all ML models"""
        try:
            all_signals = []

            # Generate signals from ML ensemble
            ml_signals = self.ml_signal_generator.generate_signals(
                current_data, options_data, ensemble_method='weighted_avg'
            )

            # Filter by confidence
            ml_signals = [signal for signal in ml_signals if signal['confidence'] >= min_confidence]
            all_signals.extend(ml_signals)

            # Generate signals from LSTM
            lstm_signals = self.lstm_generator.generate_lstm_signals(current_data)
            lstm_signals = [signal for signal in lstm_signals if signal['confidence'] >= min_confidence]
            all_signals.extend(lstm_signals)

            # Create combined signals if both models agree
            combined_signals = self._create_combined_signals(ml_signals, lstm_signals)
            all_signals.extend(combined_signals)

            return all_signals

        except Exception as e:
            self.logger.error(f"ML signal generation error: {str(e)}")
            return []

    def _create_combined_signals(self, ml_signals: List[Dict],
                               lstm_signals: List[Dict]) -> List[Dict]:
        """Create combined signals when ML and LSTM agree"""
        try:
            combined_signals = []

            # Check if both models generate signals in the same direction
            for ml_signal in ml_signals:
                for lstm_signal in lstm_signals:
                    if ml_signal['action'] == lstm_signal['action']:
                        # Create combined signal with higher confidence
                        combined_confidence = (ml_signal['confidence'] + lstm_signal['confidence']) / 2
                        combined_confidence = min(combined_confidence * 1.2, 0.95)  # Boost for agreement

                        combined_signal = {
                            'timestamp': datetime.now(),
                            'symbol': ml_signal['symbol'],
                            'action': ml_signal['action'],
                            'confidence': combined_confidence,
                            'signal_type': 'ML_LSTM_COMBINED',
                            'ml_confidence': ml_signal['confidence'],
                            'lstm_confidence': lstm_signal['confidence'],
                            'source': 'Combined ML Engine'
                        }

                        combined_signals.append(combined_signal)
                        self.logger.info(f"Combined ML signal: {ml_signal['action']} with confidence {combined_confidence:.3f}")

            return combined_signals

        except Exception as e:
            self.logger.error(f"Combined signal creation error: {str(e)}")
            return []

    def update_signal_performance(self, signal: Dict, actual_outcome: str):
        """Update performance tracking for signals"""
        try:
            signal_type = signal.get('signal_type', '').lower()

            # Determine if signal was correct
            predicted_action = signal['action']
            is_correct = (
                (predicted_action == 'BUY' and actual_outcome == 'PROFIT') or
                (predicted_action == 'SELL' and actual_outcome == 'PROFIT') or
                (predicted_action == 'HOLD' and actual_outcome == 'NEUTRAL')
            )

            # Update performance tracking
            if 'ml_ensemble' in signal_type:
                self.signal_performance['ml_ensemble']['total'] += 1
                if is_correct:
                    self.signal_performance['ml_ensemble']['correct'] += 1

            elif 'lstm' in signal_type:
                self.signal_performance['lstm']['total'] += 1
                if is_correct:
                    self.signal_performance['lstm']['correct'] += 1

            elif 'combined' in signal_type:
                self.signal_performance['combined']['total'] += 1
                if is_correct:
                    self.signal_performance['combined']['correct'] += 1

            self.logger.info(f"Updated performance for {signal_type}: {is_correct}")

        except Exception as e:
            self.logger.error(f"Performance update error: {str(e)}")

    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for all ML models"""
        try:
            metrics = {}

            for model_type, performance in self.signal_performance.items():
                if performance['total'] > 0:
                    accuracy = performance['correct'] / performance['total']
                    metrics[model_type] = {
                        'accuracy': accuracy,
                        'total_signals': performance['total'],
                        'correct_signals': performance['correct']
                    }
                else:
                    metrics[model_type] = {
                        'accuracy': 0.0,
                        'total_signals': 0,
                        'correct_signals': 0
                    }

            return metrics

        except Exception as e:
            self.logger.error(f"Performance metrics error: {str(e)}")
            return {}

    def should_retrain(self) -> bool:
        """Check if models should be retrained"""
        if self.last_training is None:
            return True

        days_since_training = (datetime.now() - self.last_training).days
        return days_since_training >= self.retrain_frequency

    def retrain_models(self, new_data: pd.DataFrame,
                      options_data: pd.DataFrame = None) -> Dict:
        """Retrain models with new data"""
        try:
            self.logger.info("Starting model retraining...")

            results = {}

            # Retrain ML ensemble
            X, y = self.ml_signal_generator.prepare_training_data(new_data, options_data)
            if len(X) > 0:
                ml_results = self.ml_signal_generator.train_models(X, y)
                results['ml_retrain'] = ml_results

            # Retrain LSTM
            lstm_results = self.lstm_generator.train_lstm(new_data)
            results['lstm_retrain'] = lstm_results

            # Save updated models
            self.save_all_models()

            self.last_training = datetime.now()
            self.logger.info("Model retraining completed")

            return results

        except Exception as e:
            self.logger.error(f"Model retraining error: {str(e)}")
            return {}

    def start_automatic_retraining(self, data_source_callback):
        """Start automatic retraining in background"""
        try:
            if self.is_running:
                self.logger.warning("Automatic retraining already running")
                return

            self.is_running = True

            def retraining_worker():
                while self.is_running:
                    try:
                        if self.should_retrain():
                            # Get fresh data from callback
                            new_data, options_data = data_source_callback()
                            if len(new_data) > 100:  # Minimum data requirement
                                self.retrain_models(new_data, options_data)

                        # Sleep for 24 hours
                        for _ in range(24 * 60):  # Check every minute for stop signal
                            if not self.is_running:
                                break
                            time.sleep(60)

                    except Exception as e:
                        self.logger.error(f"Retraining worker error: {str(e)}")
                        time.sleep(3600)  # Wait 1 hour before retry

            self.training_thread = threading.Thread(target=retraining_worker, daemon=True)
            self.training_thread.start()

            self.logger.info("Automatic retraining started")

        except Exception as e:
            self.logger.error(f"Automatic retraining start error: {str(e)}")

    def stop_automatic_retraining(self):
        """Stop automatic retraining"""
        self.is_running = False
        if self.training_thread:
            self.training_thread.join(timeout=10)
        self.logger.info("Automatic retraining stopped")

    def save_all_models(self):
        """Save all trained models"""
        try:
            self.ml_signal_generator.save_models()
            self.lstm_generator.save_lstm_model()
            self.logger.info("All ML models saved")
        except Exception as e:
            self.logger.error(f"Model saving error: {str(e)}")

    def load_all_models(self):
        """Load all trained models"""
        try:
            self.ml_signal_generator.load_models()
            self.lstm_generator.load_lstm_model()
            self.logger.info("All ML models loaded")
        except Exception as e:
            self.logger.error(f"Model loading error: {str(e)}")

    def get_feature_importance_summary(self) -> Dict:
        """Get feature importance from all models"""
        try:
            importance_data = {}

            # Get ML model importance
            ml_importance = self.ml_signal_generator.get_feature_importance()
            importance_data['ml_models'] = ml_importance

            # Get overall feature ranking
            if ml_importance:
                # Aggregate importance across models
                feature_scores = {}
                for model_name, features in ml_importance.items():
                    for feature, importance in features.items():
                        if feature not in feature_scores:
                            feature_scores[feature] = []
                        feature_scores[feature].append(importance)

                # Calculate average importance
                avg_importance = {
                    feature: np.mean(scores)
                    for feature, scores in feature_scores.items()
                }

                # Sort by importance
                sorted_features = sorted(
                    avg_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                importance_data['top_features'] = sorted_features[:20]

            return importance_data

        except Exception as e:
            self.logger.error(f"Feature importance summary error: {str(e)}")
            return {}

    def get_model_status(self) -> Dict:
        """Get status of all ML models"""
        try:
            status = {
                'ml_ensemble': {
                    'is_trained': self.ml_signal_generator.is_trained,
                    'model_count': len(self.ml_signal_generator.models),
                    'best_model': getattr(self.ml_signal_generator, 'best_model', None),
                    'performance': self.ml_signal_generator.model_performance
                },
                'lstm': {
                    'is_trained': self.lstm_generator.is_trained,
                    'sequence_length': self.lstm_generator.sequence_length,
                    'prediction_horizon': self.lstm_generator.prediction_horizon
                },
                'last_training': self.last_training,
                'retrain_frequency': self.retrain_frequency,
                'should_retrain': self.should_retrain(),
                'automatic_retraining': self.is_running,
                'performance_metrics': self.get_performance_metrics()
            }

            return status

        except Exception as e:
            self.logger.error(f"Model status error: {str(e)}")
            return {}

    def create_ml_dashboard_data(self) -> Dict:
        """Create data for ML dashboard display"""
        try:
            dashboard_data = {
                'model_status': self.get_model_status(),
                'feature_importance': self.get_feature_importance_summary(),
                'performance_metrics': self.get_performance_metrics(),
                'training_history': self.lstm_generator.get_training_plots_data()
            }

            return dashboard_data

        except Exception as e:
            self.logger.error(f"Dashboard data creation error: {str(e)}")
            return {}