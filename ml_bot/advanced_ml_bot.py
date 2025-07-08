#!/usr/bin/env python3
"""
Advanced ML Trading Bot with Web Scraping and News Analysis
Continuously learns from multiple data sources and improves predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import asyncio
import websockets
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import time
import threading
from dataclasses import dataclass, asdict

# ML and Data Processing
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Web Scraping and News Analysis
import trafilatura
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None
import yfinance as yf
import feedparser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class NewsData:
    """News data structure"""
    title: str
    content: str
    source: str
    timestamp: str
    sentiment_score: float
    market_impact: str
    symbols_mentioned: List[str]

@dataclass
class MLPrediction:
    """ML Prediction data structure"""
    symbol: str
    action: str
    confidence: float
    features: Dict
    news_sentiment: float
    web_sentiment: float
    technical_score: float
    final_score: float
    reasoning: str
    timestamp: str

class AdvancedMLBot:
    """Advanced ML Bot with Learning Capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ML Models
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Data Sources
        self.news_sources = [
            "https://feeds.feedburner.com/ndtvprofit-latest",
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "https://www.moneycontrol.com/rss/results.xml",
            "https://www.livemint.com/rss/markets"
        ]
        
        self.web_sources = [
            "https://www.screener.in",
            "https://www.tradingview.com",
            "https://in.investing.com",
            "https://www.moneycontrol.com"
        ]
        
        # Learning databases
        self.news_db = []
        self.prediction_history = []
        self.accuracy_tracking = {}
        
        # Sentiment analysis
        self.sentiment_keywords = {
            'bullish': ['up', 'rise', 'bull', 'positive', 'gain', 'surge', 'rally', 'bullish'],
            'bearish': ['down', 'fall', 'bear', 'negative', 'loss', 'crash', 'decline', 'bearish'],
            'neutral': ['stable', 'flat', 'unchanged', 'sideways', 'consolidation']
        }
        
        self.logger.info("Advanced ML Bot initialized")
    
    def scrape_news_data(self) -> List[NewsData]:
        """Scrape news from multiple sources"""
        news_data = []
        
        for source_url in self.news_sources:
            try:
                feed = feedparser.parse(source_url)
                
                for entry in feed.entries[:10]:  # Get latest 10 articles
                    try:
                        # Get full article content
                        content = trafilatura.extract(trafilatura.fetch_url(entry.link))
                        if not content:
                            content = entry.summary
                        
                        # Analyze sentiment
                        sentiment_score = self.analyze_text_sentiment(content)
                        market_impact = self.determine_market_impact(content)
                        symbols = self.extract_symbols_mentioned(content)
                        
                        news_item = NewsData(
                            title=entry.title,
                            content=content,
                            source=source_url,
                            timestamp=datetime.now().isoformat(),
                            sentiment_score=sentiment_score,
                            market_impact=market_impact,
                            symbols_mentioned=symbols
                        )
                        
                        news_data.append(news_item)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing news item: {e}")
                        continue
                        
            except Exception as e:
                self.logger.error(f"Error scraping news from {source_url}: {e}")
                continue
        
        self.news_db.extend(news_data)
        self.logger.info(f"Scraped {len(news_data)} news articles")
        return news_data
    
    def analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text content"""
        try:
            # Use TextBlob for sentiment analysis if available
            if TextBlob:
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity
            else:
                sentiment = 0.0
            
            # Enhance with keyword analysis
            text_lower = text.lower()
            bullish_count = sum(1 for word in self.sentiment_keywords['bullish'] if word in text_lower)
            bearish_count = sum(1 for word in self.sentiment_keywords['bearish'] if word in text_lower)
            
            # Combine TextBlob sentiment with keyword analysis
            keyword_sentiment = (bullish_count - bearish_count) / max(len(text.split()), 1)
            
            # Weighted average
            final_sentiment = (sentiment * 0.7) + (keyword_sentiment * 0.3)
            
            return max(-1.0, min(1.0, final_sentiment))
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return 0.0
    
    def determine_market_impact(self, content: str) -> str:
        """Determine market impact level from content"""
        content_lower = content.lower()
        
        high_impact_keywords = ['rbi', 'policy', 'rate', 'inflation', 'gdp', 'budget', 'election']
        medium_impact_keywords = ['earnings', 'results', 'ipo', 'merger', 'acquisition']
        
        high_count = sum(1 for word in high_impact_keywords if word in content_lower)
        medium_count = sum(1 for word in medium_impact_keywords if word in content_lower)
        
        if high_count >= 2:
            return 'HIGH'
        elif medium_count >= 1 or high_count == 1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def extract_symbols_mentioned(self, content: str) -> List[str]:
        """Extract trading symbols mentioned in content"""
        symbols = []
        content_upper = content.upper()
        
        common_symbols = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'SENSEX', 'RELIANCE', 'TCS', 'HDFC']
        
        for symbol in common_symbols:
            if symbol in content_upper:
                symbols.append(symbol)
        
        return symbols
    
    def scrape_web_data(self, urls: List[str]) -> Dict:
        """Scrape additional data from web sources"""
        web_data = {
            'market_mood': 'neutral',
            'analyst_sentiment': 0.0,
            'technical_indicators': {},
            'social_sentiment': 0.0
        }
        
        for url in urls:
            try:
                content = trafilatura.extract(trafilatura.fetch_url(url))
                if content:
                    sentiment = self.analyze_text_sentiment(content)
                    web_data['social_sentiment'] += sentiment
                    
            except Exception as e:
                self.logger.error(f"Error scraping {url}: {e}")
                continue
        
        # Average the sentiment
        if len(urls) > 0:
            web_data['social_sentiment'] /= len(urls)
        
        return web_data
    
    def enhance_features_with_external_data(self, base_features: Dict) -> Dict:
        """Enhance base features with external data"""
        try:
            # Get latest news sentiment
            recent_news = [n for n in self.news_db if 
                          datetime.fromisoformat(n.timestamp) > datetime.now() - timedelta(hours=6)]
            
            if recent_news:
                news_sentiment = np.mean([n.sentiment_score for n in recent_news])
                high_impact_news = len([n for n in recent_news if n.market_impact == 'HIGH'])
            else:
                news_sentiment = 0.0
                high_impact_news = 0
            
            # Get web sentiment
            web_data = self.scrape_web_data(self.web_sources[:2])  # Limit to avoid timeouts
            
            # Enhanced features
            enhanced_features = base_features.copy()
            enhanced_features.update({
                'news_sentiment': news_sentiment,
                'high_impact_news_count': high_impact_news,
                'web_sentiment': web_data['social_sentiment'],
                'market_mood_score': self.calculate_market_mood_score(),
                'volatility_from_news': abs(news_sentiment) * high_impact_news,
                'sentiment_momentum': self.calculate_sentiment_momentum()
            })
            
            return enhanced_features
            
        except Exception as e:
            self.logger.error(f"Error enhancing features: {e}")
            return base_features
    
    def calculate_market_mood_score(self) -> float:
        """Calculate overall market mood from recent data"""
        try:
            recent_news = [n for n in self.news_db if 
                          datetime.fromisoformat(n.timestamp) > datetime.now() - timedelta(hours=24)]
            
            if not recent_news:
                return 0.0
            
            sentiment_scores = [n.sentiment_score for n in recent_news]
            weighted_sentiment = np.average(sentiment_scores, 
                                          weights=[3 if n.market_impact == 'HIGH' else 
                                                 2 if n.market_impact == 'MEDIUM' else 1 
                                                 for n in recent_news])
            
            return weighted_sentiment
            
        except Exception as e:
            self.logger.error(f"Error calculating market mood: {e}")
            return 0.0
    
    def calculate_sentiment_momentum(self) -> float:
        """Calculate sentiment momentum over time"""
        try:
            now = datetime.now()
            
            # Get sentiment for last 2 hours vs previous 2 hours
            recent_news = [n for n in self.news_db if 
                          now - timedelta(hours=2) < datetime.fromisoformat(n.timestamp) < now]
            
            previous_news = [n for n in self.news_db if 
                           now - timedelta(hours=4) < datetime.fromisoformat(n.timestamp) < now - timedelta(hours=2)]
            
            if recent_news and previous_news:
                recent_sentiment = np.mean([n.sentiment_score for n in recent_news])
                previous_sentiment = np.mean([n.sentiment_score for n in previous_news])
                return recent_sentiment - previous_sentiment
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment momentum: {e}")
            return 0.0
    
    def train_enhanced_model(self, symbol: str = 'NIFTY') -> Dict:
        """Train model with enhanced features"""
        try:
            self.logger.info(f"Training enhanced model for {symbol}")
            
            # Scrape latest news data
            self.scrape_news_data()
            
            # Generate training data with enhanced features
            training_data = self.generate_enhanced_training_data(symbol)
            
            if training_data.empty:
                raise ValueError("No training data available")
            
            # Prepare features and targets
            feature_columns = [
                'rsi', 'ema_short', 'ema_long', 'vwap_ratio', 'volume_ratio',
                'price_momentum', 'volatility', 'news_sentiment', 'web_sentiment',
                'market_mood_score', 'sentiment_momentum', 'high_impact_news_count'
            ]
            
            X = training_data[feature_columns].fillna(0).values
            y = training_data['target'].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Initialize models
            self.models = {
                'random_forest': RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42),
                'svm': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
                'logistic': LogisticRegression(C=1.0, random_state=42, max_iter=1000),
                'neural_network': MLPClassifier(hidden_layer_sizes=(150, 75), max_iter=1000, random_state=42)
            }
            
            # Train models
            results = {}
            for name, model in self.models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {'accuracy': accuracy, 'model': model}
                self.logger.info(f"{name} accuracy: {accuracy:.3f}")
            
            # Create ensemble
            self.ensemble_model = VotingClassifier(
                estimators=[(name, model) for name, model in self.models.items()],
                voting='soft'
            )
            self.ensemble_model.fit(X_train, y_train)
            
            ensemble_accuracy = accuracy_score(y_test, self.ensemble_model.predict(X_test))
            results['ensemble'] = {'accuracy': ensemble_accuracy}
            
            self.is_trained = True
            self.logger.info(f"Enhanced model training completed. Ensemble accuracy: {ensemble_accuracy:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Enhanced model training error: {e}")
            raise
    
    def generate_enhanced_training_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """Generate training data with enhanced features"""
        try:
            # Base market data (simulated for demo)
            dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
            np.random.seed(42)
            
            base_price = 23500 if symbol == 'NIFTY' else 50000
            price_changes = np.random.normal(0, 0.02, days)
            prices = [base_price]
            
            for change in price_changes[1:]:
                prices.append(prices[-1] * (1 + change))
            
            # Create base features
            data = {
                'date': dates,
                'close': prices,
                'rsi': np.random.uniform(30, 70, days),
                'ema_short': [p * 0.99 for p in prices],
                'ema_long': [p * 0.98 for p in prices],
                'vwap_ratio': np.random.uniform(0.98, 1.02, days),
                'volume_ratio': np.random.uniform(0.5, 2.0, days),
                'price_momentum': np.random.uniform(-0.05, 0.05, days),
                'volatility': np.random.uniform(0.15, 0.35, days)
            }
            
            df = pd.DataFrame(data)
            
            # Add enhanced features
            for i in range(len(df)):
                # Simulate news sentiment for each day
                df.loc[i, 'news_sentiment'] = np.random.normal(0, 0.3)
                df.loc[i, 'web_sentiment'] = np.random.normal(0, 0.2)
                df.loc[i, 'market_mood_score'] = np.random.normal(0, 0.25)
                df.loc[i, 'sentiment_momentum'] = np.random.normal(0, 0.1)
                df.loc[i, 'high_impact_news_count'] = np.random.poisson(0.5)
            
            # Create target variable
            df['price_change'] = df['close'].pct_change()
            df['target'] = 0
            
            # Enhanced target creation using multiple factors
            for i in range(1, len(df)):
                price_signal = 1 if df.loc[i, 'price_change'] > 0.01 else -1 if df.loc[i, 'price_change'] < -0.01 else 0
                news_signal = 1 if df.loc[i, 'news_sentiment'] > 0.2 else -1 if df.loc[i, 'news_sentiment'] < -0.2 else 0
                volume_signal = 1 if df.loc[i, 'volume_ratio'] > 1.5 else 0
                
                # Combine signals
                combined_signal = price_signal + (news_signal * 0.5) + (volume_signal * 0.3)
                
                if combined_signal > 0.8:
                    df.loc[i, 'target'] = 1  # Buy signal
                elif combined_signal < -0.8:
                    df.loc[i, 'target'] = -1  # Sell signal
                else:
                    df.loc[i, 'target'] = 0  # Hold signal
            
            return df.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced training data: {e}")
            return pd.DataFrame()
    
    def generate_enhanced_prediction(self, symbol: str, current_data: Dict) -> MLPrediction:
        """Generate prediction with enhanced features"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained yet")
            
            # Enhance features with external data
            enhanced_features = self.enhance_features_with_external_data(current_data)
            
            # Prepare features for prediction
            feature_columns = [
                'rsi', 'ema_short', 'ema_long', 'vwap_ratio', 'volume_ratio',
                'price_momentum', 'volatility', 'news_sentiment', 'web_sentiment',
                'market_mood_score', 'sentiment_momentum', 'high_impact_news_count'
            ]
            
            features = [enhanced_features.get(col, 0) for col in feature_columns]
            features_scaled = self.scaler.transform([features])
            
            # Get prediction
            prediction = self.ensemble_model.predict(features_scaled)[0]
            probabilities = self.ensemble_model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
            
            # Calculate component scores
            technical_score = self.calculate_technical_score(current_data)
            news_sentiment = enhanced_features.get('news_sentiment', 0)
            web_sentiment = enhanced_features.get('web_sentiment', 0)
            
            # Final weighted score
            final_score = (technical_score * 0.4) + (news_sentiment * 0.3) + (web_sentiment * 0.3)
            
            # Determine action
            if prediction == 1 and confidence > 0.65:
                action = 'BUY_CE'
                reasoning = f"Bullish: Tech={technical_score:.2f}, News={news_sentiment:.2f}, Web={web_sentiment:.2f}"
            elif prediction == -1 and confidence > 0.65:
                action = 'BUY_PE'
                reasoning = f"Bearish: Tech={technical_score:.2f}, News={news_sentiment:.2f}, Web={web_sentiment:.2f}"
            else:
                action = 'WAIT'
                reasoning = f"Low confidence: {confidence:.1%} or conflicting signals"
            
            # Create prediction object
            ml_prediction = MLPrediction(
                symbol=symbol,
                action=action,
                confidence=confidence,
                features=enhanced_features,
                news_sentiment=news_sentiment,
                web_sentiment=web_sentiment,
                technical_score=technical_score,
                final_score=final_score,
                reasoning=reasoning,
                timestamp=datetime.now().isoformat()
            )
            
            # Store for learning
            self.prediction_history.append(ml_prediction)
            
            return ml_prediction
            
        except Exception as e:
            self.logger.error(f"Enhanced prediction error: {e}")
            raise
    
    def calculate_technical_score(self, data: Dict) -> float:
        """Calculate technical analysis score"""
        try:
            rsi = data.get('rsi', 50)
            ema_short = data.get('ema_short', 0)
            ema_long = data.get('ema_long', 0)
            volume_ratio = data.get('volume_ratio', 1)
            
            # RSI score
            if rsi > 70:
                rsi_score = -0.5  # Overbought
            elif rsi < 30:
                rsi_score = 0.5   # Oversold
            elif rsi > 55:
                rsi_score = 0.3   # Bullish
            elif rsi < 45:
                rsi_score = -0.3  # Bearish
            else:
                rsi_score = 0     # Neutral
            
            # EMA score
            if ema_short > ema_long:
                ema_score = 0.4
            else:
                ema_score = -0.4
            
            # Volume score
            volume_score = min(0.3, (volume_ratio - 1) * 0.3)
            
            return rsi_score + ema_score + volume_score
            
        except Exception as e:
            self.logger.error(f"Technical score calculation error: {e}")
            return 0.0
    
    def update_learning_from_feedback(self, prediction_id: str, actual_result: str):
        """Update learning based on prediction results"""
        try:
            # Find the prediction
            prediction = None
            for pred in self.prediction_history:
                if pred.timestamp == prediction_id:
                    prediction = pred
                    break
            
            if not prediction:
                return
            
            # Update accuracy tracking
            symbol = prediction.symbol
            if symbol not in self.accuracy_tracking:
                self.accuracy_tracking[symbol] = {
                    'correct': 0,
                    'total': 0,
                    'accuracy': 0.0
                }
            
            # Check if prediction was correct
            is_correct = (
                (prediction.action == 'BUY_CE' and actual_result == 'PROFIT') or
                (prediction.action == 'BUY_PE' and actual_result == 'PROFIT') or
                (prediction.action == 'WAIT' and actual_result == 'NO_CHANGE')
            )
            
            if is_correct:
                self.accuracy_tracking[symbol]['correct'] += 1
            
            self.accuracy_tracking[symbol]['total'] += 1
            self.accuracy_tracking[symbol]['accuracy'] = (
                self.accuracy_tracking[symbol]['correct'] / 
                self.accuracy_tracking[symbol]['total']
            )
            
            self.logger.info(f"Learning updated for {symbol}: {self.accuracy_tracking[symbol]['accuracy']:.1%} accuracy")
            
        except Exception as e:
            self.logger.error(f"Learning update error: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for all symbols"""
        return {
            'accuracy_by_symbol': self.accuracy_tracking,
            'total_predictions': len(self.prediction_history),
            'news_articles_processed': len(self.news_db),
            'last_training': datetime.now().isoformat()
        }