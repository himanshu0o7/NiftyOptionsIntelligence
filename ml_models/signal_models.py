import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Use simpler alternatives to avoid OpenMP dependency issues
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
import joblib
import os
from datetime import datetime
from utils.logger import Logger
from .feature_engineering import FeatureEngineer

class MLSignalGenerator:
    """Advanced ML models for generating trading signals"""
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.logger = Logger()
        self.is_trained = False
        self.model_performance = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different ML models"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=50,  # Reduced for performance
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=1  # Avoid OpenMP issues
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=50,  # Reduced for performance
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='liblinear'  # More stable solver
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                gamma='scale'
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(50, 25),  # Reduced complexity
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        }
        
        # Add XGBoost if available
        if XGB_AVAILABLE:
            try:
                self.models['xgboost'] = xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    nthread=1  # Single thread to avoid OpenMP
                )
            except Exception as e:
                self.logger.warning(f"XGBoost initialization failed: {str(e)}")
        
        # Add LightGBM if available
        if LGB_AVAILABLE:
            try:
                self.models['lightgbm'] = lgb.LGBMClassifier(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                    num_threads=1  # Single thread to avoid OpenMP
                )
            except Exception as e:
                self.logger.warning(f"LightGBM initialization failed: {str(e)}")
    
    def _create_deep_learning_model(self, input_shape: int):
        """Create deep learning model using TensorFlow/Keras"""
        if not TF_AVAILABLE:
            self.logger.warning("TensorFlow not available, skipping deep learning model")
            return None
            
        try:
            model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(input_shape,)),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(3, activation='softmax')  # BUY, SELL, HOLD
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        except Exception as e:
            self.logger.error(f"Deep learning model creation failed: {str(e)}")
            return None
    
    def prepare_training_data(self, market_data: pd.DataFrame, 
                            options_data: pd.DataFrame = None,
                            lookback_period: int = 20) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with features and labels"""
        try:
            # Prepare features
            features_df = self.feature_engineer.prepare_features(market_data, options_data)
            
            # Create labels based on future price movements
            features_df['future_return'] = features_df['close'].shift(-lookback_period) / features_df['close'] - 1
            
            # Define signal labels: 0=SELL, 1=HOLD, 2=BUY
            def create_signal_labels(returns, buy_threshold=0.02, sell_threshold=-0.02):
                labels = []
                for ret in returns:
                    if pd.isna(ret):
                        labels.append(1)  # HOLD for NaN
                    elif ret > buy_threshold:
                        labels.append(2)  # BUY
                    elif ret < sell_threshold:
                        labels.append(0)  # SELL
                    else:
                        labels.append(1)  # HOLD
                return labels
            
            features_df['signal_label'] = create_signal_labels(features_df['future_return'])
            
            # Remove rows with NaN in future returns and recent rows
            features_df = features_df[:-lookback_period]
            features_df = features_df.dropna()
            
            # Separate features and labels
            feature_cols = self.feature_engineer.feature_names
            X = features_df[feature_cols]
            y = features_df['signal_label']
            
            self.logger.info(f"Prepared training data: {len(X)} samples, {len(feature_cols)} features")
            self.logger.info(f"Label distribution: {y.value_counts().to_dict()}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Training data preparation error: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2, validate: bool = True) -> Dict:
        """Train all ML models"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.feature_engineer.scale_features(X_train, fit=True)
            X_test_scaled = self.feature_engineer.scale_features(X_test, fit=False)
            
            training_results = {}
            
            # Train each model
            for name, model in self.models.items():
                try:
                    self.logger.info(f"Training {name} model...")
                    
                    # Use scaled features for models that benefit from scaling
                    if name in ['logistic_regression', 'svm', 'neural_network']:
                        model.fit(X_train_scaled[self.feature_engineer.feature_names], y_train)
                        if validate:
                            y_pred = model.predict(X_test_scaled[self.feature_engineer.feature_names])
                            accuracy = accuracy_score(y_test, y_pred)
                    else:
                        model.fit(X_train[self.feature_engineer.feature_names], y_train)
                        if validate:
                            y_pred = model.predict(X_test[self.feature_engineer.feature_names])
                            accuracy = accuracy_score(y_test, y_pred)
                    
                    if validate:
                        training_results[name] = {
                            'accuracy': accuracy,
                            'classification_report': classification_report(y_test, y_pred)
                        }
                        self.logger.info(f"{name} accuracy: {accuracy:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {name}: {str(e)}")
                    continue
            
            # Train deep learning model if TensorFlow is available
            if TF_AVAILABLE:
                try:
                    self.logger.info("Training deep learning model...")
                    dl_model = self._create_deep_learning_model(len(self.feature_engineer.feature_names))
                    
                    if dl_model is not None:
                        # Prepare data for DL model
                        X_train_dl = X_train_scaled[self.feature_engineer.feature_names].values
                        X_test_dl = X_test_scaled[self.feature_engineer.feature_names].values
                        
                        # Train with early stopping
                        early_stopping = keras.callbacks.EarlyStopping(
                            monitor='val_loss', patience=5, restore_best_weights=True
                        )
                        
                        history = dl_model.fit(
                            X_train_dl, y_train,
                            epochs=50,  # Reduced epochs
                            batch_size=32,
                            validation_split=0.2,
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        
                        if validate:
                            y_pred_dl = dl_model.predict(X_test_dl, verbose=0)
                            y_pred_dl_classes = np.argmax(y_pred_dl, axis=1)
                            dl_accuracy = accuracy_score(y_test, y_pred_dl_classes)
                            
                            training_results['deep_learning'] = {
                                'accuracy': dl_accuracy,
                                'classification_report': classification_report(y_test, y_pred_dl_classes)
                            }
                            self.logger.info(f"Deep learning accuracy: {dl_accuracy:.4f}")
                        
                        self.models['deep_learning'] = dl_model
                    
                except Exception as e:
                    self.logger.error(f"Error training deep learning model: {str(e)}")
            else:
                self.logger.info("TensorFlow not available, skipping deep learning model")
            
            self.model_performance = training_results
            self.is_trained = True
            
            # Save best performing model
            best_model_name = max(training_results.keys(), key=lambda k: training_results[k]['accuracy'])
            self.best_model = best_model_name
            
            self.logger.info(f"Training completed. Best model: {best_model_name}")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Model training error: {str(e)}")
            return {}
    
    def hyperparameter_optimization(self, X: pd.DataFrame, y: pd.Series, 
                                  model_name: str = 'xgboost') -> Dict:
        """Optimize hyperparameters for selected model"""
        try:
            if model_name == 'xgboost':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
                model = xgb.XGBClassifier(random_state=42)
                
            elif model_name == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                model = RandomForestClassifier(random_state=42)
            
            else:
                self.logger.warning(f"Hyperparameter optimization not implemented for {model_name}")
                return {}
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            
            grid_search.fit(X[self.feature_engineer.feature_names], y)
            
            # Update model with best parameters
            self.models[model_name] = grid_search.best_estimator_
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            self.logger.info(f"Hyperparameter optimization completed for {model_name}")
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization error: {str(e)}")
            return {}
    
    def generate_signals(self, current_data: pd.DataFrame, 
                        options_data: pd.DataFrame = None,
                        ensemble_method: str = 'voting') -> List[Dict]:
        """Generate trading signals using trained ML models"""
        try:
            if not self.is_trained:
                self.logger.warning("Models not trained yet. Cannot generate signals.")
                return []
            
            # Prepare features
            features_df = self.feature_engineer.prepare_features(current_data, options_data)
            
            # Get the latest data point
            latest_features = features_df.iloc[-1:][self.feature_engineer.feature_names]
            
            # Scale features
            latest_features_scaled = self.feature_engineer.scale_features(latest_features, fit=False)
            
            signals = []
            model_predictions = {}
            
            # Get predictions from each model
            for name, model in self.models.items():
                try:
                    if name == 'deep_learning':
                        if hasattr(model, 'predict'):
                            pred_proba = model.predict(latest_features_scaled[self.feature_engineer.feature_names].values)
                            pred_class = np.argmax(pred_proba, axis=1)[0]
                            confidence = np.max(pred_proba)
                        else:
                            continue
                    else:
                        # Use scaled features for models that benefit from scaling
                        if name in ['logistic_regression', 'svm', 'neural_network']:
                            pred_class = model.predict(latest_features_scaled[self.feature_engineer.feature_names])[0]
                            if hasattr(model, 'predict_proba'):
                                pred_proba = model.predict_proba(latest_features_scaled[self.feature_engineer.feature_names])[0]
                                confidence = np.max(pred_proba)
                            else:
                                confidence = 0.5
                        else:
                            pred_class = model.predict(latest_features[self.feature_engineer.feature_names])[0]
                            if hasattr(model, 'predict_proba'):
                                pred_proba = model.predict_proba(latest_features[self.feature_engineer.feature_names])[0]
                                confidence = np.max(pred_proba)
                            else:
                                confidence = 0.5
                    
                    model_predictions[name] = {
                        'prediction': pred_class,
                        'confidence': confidence
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error getting prediction from {name}: {str(e)}")
                    continue
            
            # Generate ensemble signal
            if ensemble_method == 'voting':
                # Majority voting
                predictions = [pred['prediction'] for pred in model_predictions.values()]
                if predictions:
                    ensemble_prediction = max(set(predictions), key=predictions.count)
                    ensemble_confidence = predictions.count(ensemble_prediction) / len(predictions)
                else:
                    ensemble_prediction = 1  # HOLD
                    ensemble_confidence = 0.5
                    
            elif ensemble_method == 'weighted_avg':
                # Weighted by model performance
                total_weight = 0
                weighted_sum = 0
                
                for name, pred in model_predictions.items():
                    if name in self.model_performance:
                        weight = self.model_performance[name]['accuracy']
                        weighted_sum += pred['prediction'] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    ensemble_prediction = round(weighted_sum / total_weight)
                    ensemble_confidence = total_weight / len(model_predictions)
                else:
                    ensemble_prediction = 1  # HOLD
                    ensemble_confidence = 0.5
            
            # Convert prediction to signal
            signal_mapping = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            signal_action = signal_mapping.get(ensemble_prediction, 'HOLD')
            
            # Create signal if not HOLD and confidence is sufficient
            if signal_action != 'HOLD' and ensemble_confidence > 0.6:
                signal = {
                    'timestamp': datetime.now(),
                    'symbol': 'NIFTY',  # This would come from input data
                    'action': signal_action,
                    'confidence': ensemble_confidence,
                    'signal_type': 'ML_ENSEMBLE',
                    'model_predictions': model_predictions,
                    'features_used': len(self.feature_engineer.feature_names),
                    'source': 'ML Signal Generator'
                }
                signals.append(signal)
                
                self.logger.info(f"Generated ML signal: {signal_action} with confidence {ensemble_confidence:.3f}")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {str(e)}")
            return []
    
    def get_feature_importance(self, model_name: str = None) -> Dict:
        """Get feature importance from trained models"""
        try:
            if not self.is_trained:
                return {}
            
            importance_data = {}
            
            models_to_check = [model_name] if model_name else self.models.keys()
            
            for name in models_to_check:
                if name not in self.models:
                    continue
                    
                model = self.models[name]
                
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    importance_data[name] = dict(zip(
                        self.feature_engineer.feature_names,
                        model.feature_importances_
                    ))
                elif hasattr(model, 'coef_'):
                    # Linear models
                    if len(model.coef_.shape) > 1:
                        # Multi-class
                        importance_data[name] = dict(zip(
                            self.feature_engineer.feature_names,
                            np.abs(model.coef_).mean(axis=0)
                        ))
                    else:
                        importance_data[name] = dict(zip(
                            self.feature_engineer.feature_names,
                            np.abs(model.coef_)
                        ))
            
            return importance_data
            
        except Exception as e:
            self.logger.error(f"Feature importance error: {str(e)}")
            return {}
    
    def save_models(self, filepath: str = 'ml_models/saved_models'):
        """Save trained models to disk"""
        try:
            os.makedirs(filepath, exist_ok=True)
            
            for name, model in self.models.items():
                if name == 'deep_learning':
                    model.save(f"{filepath}/{name}_model.h5")
                else:
                    joblib.dump(model, f"{filepath}/{name}_model.joblib")
            
            # Save feature engineer
            joblib.dump(self.feature_engineer, f"{filepath}/feature_engineer.joblib")
            
            # Save metadata
            metadata = {
                'is_trained': self.is_trained,
                'model_performance': self.model_performance,
                'best_model': getattr(self, 'best_model', None),
                'feature_names': self.feature_engineer.feature_names
            }
            joblib.dump(metadata, f"{filepath}/metadata.joblib")
            
            self.logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model saving error: {str(e)}")
    
    def load_models(self, filepath: str = 'ml_models/saved_models'):
        """Load trained models from disk"""
        try:
            # Load metadata
            metadata = joblib.load(f"{filepath}/metadata.joblib")
            self.is_trained = metadata['is_trained']
            self.model_performance = metadata['model_performance']
            self.best_model = metadata.get('best_model', None)
            
            # Load feature engineer
            self.feature_engineer = joblib.load(f"{filepath}/feature_engineer.joblib")
            
            # Load models
            for name in self.models.keys():
                try:
                    if name == 'deep_learning':
                        if os.path.exists(f"{filepath}/{name}_model.h5"):
                            self.models[name] = keras.models.load_model(f"{filepath}/{name}_model.h5")
                    else:
                        if os.path.exists(f"{filepath}/{name}_model.joblib"):
                            self.models[name] = joblib.load(f"{filepath}/{name}_model.joblib")
                except Exception as e:
                    self.logger.warning(f"Could not load {name} model: {str(e)}")
                    continue
            
            self.logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model loading error: {str(e)}")
    
    def get_model_summary(self) -> Dict:
        """Get summary of all models and their performance"""
        try:
            summary = {
                'is_trained': self.is_trained,
                'num_models': len(self.models),
                'feature_count': len(self.feature_engineer.feature_names),
                'model_performance': self.model_performance,
                'best_model': getattr(self, 'best_model', None)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Model summary error: {str(e)}")
            return {}