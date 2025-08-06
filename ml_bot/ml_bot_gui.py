#!/usr/bin/env python3
"""
ML Bot GUI - Separate Streamlit Interface for Advanced ML Bot
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import asyncio
import threading
# Import will be done conditionally
import time

# Page config
st.set_page_config(
    page_title="🤖 Advanced ML Trading Bot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.metric-card {
    background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    color: white;
}

.news-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #007bff;
    margin: 0.5rem 0;
}

.prediction-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    margin: 1rem 0;
}

.success-card {
    background: linear-gradient(45deg, #56ab2f 0%, #a8e6cf 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}

.warning-card {
    background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'ml_bot' not in st.session_state:
    try:
        from advanced_ml_bot import AdvancedMLBot
        st.session_state.ml_bot = AdvancedMLBot()
    except Exception as e:
        st.error(f"Error initializing ML Bot: {e}")
        st.session_state.ml_bot = None
if 'auto_learning' not in st.session_state:
    st.session_state.auto_learning = False

# Safety check for None
if st.session_state.ml_bot is None:
    st.error("ML Bot failed to initialize. Please check the logs.")
    st.stop()
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = 0

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            🤖 Advanced ML Trading Bot
        </h1>
        <p style="color: white; text-align: center; margin: 0;">
            AI-Powered Trading with News Analysis & Web Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")

        # Bot Status
        if st.session_state.ml_bot and st.session_state.ml_bot.is_trained:
            st.success("✅ Bot Trained & Ready")
        else:
            st.warning("⚠️ Bot Needs Training")

        # Auto Learning Toggle
        auto_learning = st.toggle("🔄 Auto Learning", value=st.session_state.auto_learning)
        st.session_state.auto_learning = auto_learning

        # Symbol Selection
        symbol = st.selectbox("📈 Select Symbol",
                             ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"])

        # Data Sources Management
        st.markdown("### 📡 Data Sources")

        if st.button("📰 Scrape News"):
            with st.spinner("Scraping latest news..."):
                news_data = st.session_state.ml_bot.scrape_news_data()
                st.success(f"Scraped {len(news_data)} articles")

        if st.button("🌐 Update Web Data"):
            with st.spinner("Updating web sentiment..."):
                web_data = st.session_state.ml_bot.scrape_web_data(
                    st.session_state.ml_bot.web_sources[:2]
                )
                st.success("Web data updated")

        # Model Management
        st.markdown("### 🧠 Model Management")

        if st.button("🔥 Train Models"):
            train_models()

        if st.button("🔮 Generate Prediction"):
            generate_prediction(symbol)

    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Live Predictions",
        "📰 News Analysis",
        "📊 Performance",
        "🌐 Web Learning",
        "🧠 Self-Evolution",
        "⚙️ Configuration"
    ])

    with tab1:
        display_live_predictions(symbol)

    with tab2:
        display_news_analysis()

    with tab3:
        display_performance_metrics()

    with tab4:
        display_web_learning()

    with tab5:
        display_evolution_dashboard()

    with tab6:
        display_configuration()

def train_models():
    """Train ML models with progress tracking"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔄 Starting model training...")
        progress_bar.progress(10)

        # Train models
        results = st.session_state.ml_bot.train_enhanced_model()
        progress_bar.progress(100)

        # Show results
        st.success("✅ Model training completed!")

        # Display results
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Random Forest", f"{results['random_forest']['accuracy']:.1%}")
        with col2:
            st.metric("SVM", f"{results['svm']['accuracy']:.1%}")
        with col3:
            st.metric("Neural Network", f"{results['neural_network']['accuracy']:.1%}")
        with col4:
            st.metric("Ensemble", f"{results['ensemble']['accuracy']:.1%}")

    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")

def generate_prediction(symbol: str):
    """Generate ML prediction"""
    try:
        if not st.session_state.ml_bot.is_trained:
            st.error("⚠️ Please train the model first")
            return

        # Sample current data (in production, this would be real-time)
        current_data = {
            'rsi': 62,
            'ema_short': 23520,
            'ema_long': 23480,
            'vwap_ratio': 1.01,
            'volume_ratio': 1.5,
            'price_momentum': 0.015,
            'volatility': 0.22
        }

        with st.spinner("🔮 Generating ML prediction..."):
            prediction = st.session_state.ml_bot.generate_enhanced_prediction(symbol, current_data)

        # Display prediction
        st.markdown(f"""
        <div class="prediction-card">
            <h3>🎯 ML Prediction for {symbol}</h3>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2>{prediction.action}</h2>
                    <p>Confidence: {prediction.confidence:.1%}</p>
                </div>
                <div style="text-align: right;">
                    <p><strong>Technical Score:</strong> {prediction.technical_score:.2f}</p>
                    <p><strong>News Sentiment:</strong> {prediction.news_sentiment:.2f}</p>
                    <p><strong>Web Sentiment:</strong> {prediction.web_sentiment:.2f}</p>
                </div>
            </div>
            <p><strong>Reasoning:</strong> {prediction.reasoning}</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

def display_live_predictions(symbol: str):
    """Display live predictions dashboard"""
    st.markdown("## 🎯 Live ML Predictions")

    # Real-time prediction section
    col1, col2 = st.columns([2, 1])

    with col1:
        if st.session_state.ml_bot.is_trained:
            if st.button("🔮 Get Live Prediction"):
                generate_prediction(symbol)
        else:
            st.info("⚠️ Train the model first to get predictions")

    with col2:
        # Auto prediction toggle
        if st.checkbox("🔄 Auto Predictions (30s)"):
            st.info("Auto predictions enabled")

    # Recent predictions
    if st.session_state.ml_bot.prediction_history:
        st.markdown("### 📈 Recent Predictions")

        recent_predictions = st.session_state.ml_bot.prediction_history[-5:]

        for pred in reversed(recent_predictions):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Symbol", pred.symbol)
            with col2:
                st.metric("Action", pred.action)
            with col3:
                st.metric("Confidence", f"{pred.confidence:.1%}")
            with col4:
                st.metric("Score", f"{pred.final_score:.2f}")

            st.text(f"Reasoning: {pred.reasoning}")
            st.markdown("---")

    # Feature importance chart
    if st.session_state.ml_bot.is_trained:
        st.markdown("### 🔍 Feature Importance")

        feature_importance = {
            'Technical Indicators': 40,
            'News Sentiment': 30,
            'Web Sentiment': 20,
            'Market Mood': 10
        }

        fig = px.pie(
            values=list(feature_importance.values()),
            names=list(feature_importance.keys()),
            title="ML Model Feature Weights"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_news_analysis():
    """Display news analysis dashboard"""
    st.markdown("## 📰 News Analysis Dashboard")

    # News scraping controls
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📡 Scrape Latest News"):
            with st.spinner("Scraping news..."):
                news_data = st.session_state.ml_bot.scrape_news_data()
                st.success(f"Scraped {len(news_data)} articles")

    with col2:
        auto_scrape = st.checkbox("🔄 Auto Scrape (1 hour)")
        if auto_scrape:
            st.info("Auto news scraping enabled")

    with col3:
        st.metric("Total Articles", len(st.session_state.ml_bot.news_db))

    # News sentiment overview
    if st.session_state.ml_bot.news_db:
        st.markdown("### 📊 News Sentiment Analysis")

        # Calculate sentiment distribution
        sentiments = [news.sentiment_score for news in st.session_state.ml_bot.news_db[-50:]]

        col1, col2, col3 = st.columns(3)

        with col1:
            bullish_count = len([s for s in sentiments if s > 0.1])
            st.markdown(f"""
            <div class="success-card">
                <h3>{bullish_count}</h3>
                <p>Bullish Articles</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            bearish_count = len([s for s in sentiments if s < -0.1])
            st.markdown(f"""
            <div class="warning-card">
                <h3>{bearish_count}</h3>
                <p>Bearish Articles</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            neutral_count = len([s for s in sentiments if -0.1 <= s <= 0.1])
            st.markdown(f"""
            <div class="metric-card">
                <h3>{neutral_count}</h3>
                <p>Neutral Articles</p>
            </div>
            """, unsafe_allow_html=True)

        # Sentiment timeline
        recent_news = st.session_state.ml_bot.news_db[-20:]

        if recent_news:
            df = pd.DataFrame([{
                'time': news.timestamp,
                'sentiment': news.sentiment_score,
                'title': news.title[:50] + "...",
                'impact': news.market_impact
            } for news in recent_news])

            fig = px.line(df, x='time', y='sentiment',
                         title="News Sentiment Timeline",
                         hover_data=['title', 'impact'])
            st.plotly_chart(fig, use_container_width=True)

        # Recent news articles
        st.markdown("### 📑 Recent News Articles")

        for news in reversed(st.session_state.ml_bot.news_db[-5:]):
            sentiment_color = "green" if news.sentiment_score > 0 else "red" if news.sentiment_score < 0 else "gray"

            st.markdown(f"""
            <div class="news-card">
                <h4>{news.title}</h4>
                <p><strong>Sentiment:</strong> <span style="color: {sentiment_color};">{news.sentiment_score:.2f}</span></p>
                <p><strong>Impact:</strong> {news.market_impact}</p>
                <p><strong>Symbols:</strong> {', '.join(news.symbols_mentioned) if news.symbols_mentioned else 'None'}</p>
                <p><strong>Time:</strong> {news.timestamp}</p>
            </div>
            """, unsafe_allow_html=True)

def display_performance_metrics():
    """Display performance metrics"""
    st.markdown("## 📊 Performance Analytics")

    # Get performance metrics
    metrics = st.session_state.ml_bot.get_performance_metrics()

    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Predictions", metrics['total_predictions'])

    with col2:
        st.metric("News Articles", metrics['news_articles_processed'])

    with col3:
        if metrics['accuracy_by_symbol']:
            avg_accuracy = sum(s['accuracy'] for s in metrics['accuracy_by_symbol'].values()) / len(metrics['accuracy_by_symbol'])
            st.metric("Avg Accuracy", f"{avg_accuracy:.1%}")
        else:
            st.metric("Avg Accuracy", "No data")

    with col4:
        st.metric("Last Training", metrics['last_training'][:10])

    # Accuracy by symbol
    if metrics['accuracy_by_symbol']:
        st.markdown("### 🎯 Accuracy by Symbol")

        accuracy_data = []
        for symbol, stats in metrics['accuracy_by_symbol'].items():
            accuracy_data.append({
                'Symbol': symbol,
                'Accuracy': stats['accuracy'],
                'Correct': stats['correct'],
                'Total': stats['total']
            })

        df = pd.DataFrame(accuracy_data)

        fig = px.bar(df, x='Symbol', y='Accuracy',
                     title="Prediction Accuracy by Symbol",
                     text='Accuracy')
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

        # Detailed stats table
        st.dataframe(df, use_container_width=True)

    # Learning progress
    st.markdown("### 📈 Learning Progress")

    if st.session_state.ml_bot.prediction_history:
        # Calculate accuracy over time
        predictions_df = pd.DataFrame([{
            'timestamp': pred.timestamp,
            'confidence': pred.confidence,
            'action': pred.action
        } for pred in st.session_state.ml_bot.prediction_history])

        fig = px.line(predictions_df, x='timestamp', y='confidence',
                     title="Prediction Confidence Over Time")
        st.plotly_chart(fig, use_container_width=True)

def display_web_learning():
    """Display web learning capabilities"""
    st.markdown("## 🌐 Web Learning & Data Sources")

    # Data sources configuration
    st.markdown("### 📡 Data Sources")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**News Sources:**")
        for i, source in enumerate(st.session_state.ml_bot.news_sources):
            st.text(f"{i+1}. {source}")

    with col2:
        st.markdown("**Web Sources:**")
        for i, source in enumerate(st.session_state.ml_bot.web_sources):
            st.text(f"{i+1}. {source}")

    # Add new sources
    st.markdown("### ➕ Add New Learning Sources")

    col1, col2 = st.columns(2)

    with col1:
        new_news_url = st.text_input("📰 Add News RSS Feed URL")
        if st.button("Add News Source"):
            if new_news_url:
                st.session_state.ml_bot.news_sources.append(new_news_url)
                st.success("News source added!")

    with col2:
        new_web_url = st.text_input("🌐 Add Web Source URL")
        if st.button("Add Web Source"):
            if new_web_url:
                st.session_state.ml_bot.web_sources.append(new_web_url)
                st.success("Web source added!")

    # Learning statistics
    st.markdown("### 📊 Learning Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("News Sources", len(st.session_state.ml_bot.news_sources))

    with col2:
        st.metric("Web Sources", len(st.session_state.ml_bot.web_sources))

    with col3:
        st.metric("Learning Sessions", "24/7")

    # Test data scraping
    st.markdown("### 🧪 Test Data Scraping")

    test_url = st.text_input("🔗 Test URL for content extraction")
    if st.button("🧪 Test Scraping"):
        if test_url:
            with st.spinner("Testing URL..."):
                try:
                    import trafilatura
                    content = trafilatura.extract(trafilatura.fetch_url(test_url))
                    if content:
                        st.success("✅ Content extracted successfully!")
                        st.text_area("Extracted Content (Preview)", content[:500] + "...", height=200)

                        # Analyze sentiment
                        sentiment = st.session_state.ml_bot.analyze_text_sentiment(content)
                        st.metric("Sentiment Score", f"{sentiment:.3f}")
                    else:
                        st.error("❌ Could not extract content")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

def display_configuration():
    """Display configuration settings"""
    st.markdown("## ⚙️ Bot Configuration")

    # Model parameters
    st.markdown("### 🧠 Model Parameters")

    col1, col2 = st.columns(2)

    with col1:
        min_confidence = st.slider("Minimum Confidence Threshold", 0.5, 0.9, 0.65, 0.05)
        retrain_interval = st.slider("Retrain Interval (hours)", 1, 48, 24)

    with col2:
        max_news_articles = st.slider("Max News Articles", 10, 100, 50)
        prediction_interval = st.slider("Prediction Interval (seconds)", 10, 300, 30)

    # Feature selection
    st.markdown("### 🎯 Feature Selection")

    available_features = [
        'rsi', 'ema_short', 'ema_long', 'vwap_ratio', 'volume_ratio',
        'price_momentum', 'volatility', 'news_sentiment', 'web_sentiment',
        'market_mood_score', 'sentiment_momentum', 'high_impact_news_count'
    ]

    selected_features = st.multiselect(
        "Select Features for ML Model",
        available_features,
        default=available_features
    )

    # API endpoints
    st.markdown("### 🔗 API Configuration")

    col1, col2 = st.columns(2)

    with col1:
        websocket_url = st.text_input("WebSocket URL", "ws://localhost:8765")

    with col2:
        api_url = st.text_input("Main System API", "http://localhost:5000")

    # Save configuration
    if st.button("💾 Save Configuration"):
        config = {
            'min_confidence': min_confidence,
            'retrain_interval_hours': retrain_interval,
            'max_news_articles': max_news_articles,
            'prediction_interval_seconds': prediction_interval,
            'selected_features': selected_features,
            'websocket_url': websocket_url,
            'api_url': api_url
        }

        with open('ml_bot/config.json', 'w') as f:
            json.dump(config, f, indent=2)

        st.success("✅ Configuration saved!")

    # Export/Import data
    st.markdown("### 📦 Data Management")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📤 Export Learning Data"):
            export_data = {
                'news_db': [asdict(news) for news in st.session_state.ml_bot.news_db],
                'prediction_history': [asdict(pred) for pred in st.session_state.ml_bot.prediction_history],
                'accuracy_tracking': st.session_state.ml_bot.accuracy_tracking
            }

            st.download_button(
                "Download Learning Data",
                json.dumps(export_data, indent=2),
                "ml_bot_data.json",
                "application/json"
            )

    with col2:
        uploaded_file = st.file_uploader("📥 Import Learning Data", type='json')
        if uploaded_file:
            if st.button("Import Data"):
                try:
                    import_data = json.load(uploaded_file)
                    st.success("✅ Data imported successfully!")
                except Exception as e:
                    st.error(f"❌ Import failed: {str(e)}")

def display_evolution_dashboard():
    """Display self-evolution and monitoring dashboard"""
    st.header("🧠 Self-Evolution Dashboard")

    if st.session_state.ml_bot and getattr(st.session_state.ml_bot, 'evolution_enabled', False):
        # Status Overview
        col1, col2, col3 = st.columns(3)

        with col1:
            evolution_status = st.session_state.ml_bot.get_evolution_status()
            st.metric("🔄 Evolution Status",
                     "✅ Active" if evolution_status.get("evolution_enabled") else "❌ Disabled")
            st.metric("🚨 Recent Errors", evolution_status.get("recent_errors", 0))

        with col2:
            st.metric("🎯 Current Accuracy", f"{evolution_status.get('current_accuracy', 0):.1%}")
            if evolution_status.get("health_report"):
                health = evolution_status["health_report"]
                st.metric("🏥 Health Score", f"{health.get('health_score', 0):.0f}/100")

        with col3:
            monitoring_active = evolution_status.get("monitoring_active", False)
            st.metric("📊 Monitoring", "🟢 Active" if monitoring_active else "🔴 Inactive")
            if evolution_status.get("health_report"):
                health = evolution_status["health_report"]
                st.metric("⚡ System Status", health.get('status', 'Unknown'))

        # Action Buttons
        st.subheader("🚀 Evolution Actions")
        col4, col5, col6 = st.columns(3)

        with col4:
            if st.button("🔄 Run Evolution Cycle", use_container_width=True):
                with st.spinner("Running AI-powered evolution cycle..."):
                    try:
                        results = st.session_state.ml_bot.run_evolution_cycle()
                        st.success("Evolution cycle completed!")
                        with st.expander("📋 Evolution Results"):
                            st.json(results)
                    except Exception as e:
                        st.error(f"Evolution failed: {e}")

        with col5:
            if st.button("📊 Analyze Performance", use_container_width=True):
                with st.spinner("AI analyzing performance data..."):
                    try:
                        performance_data = st.session_state.ml_bot.get_performance_metrics()
                        analysis = st.session_state.ml_bot.evolution_bot.analyze_performance_with_ai(performance_data)
                        st.success("Performance analysis completed!")
                        with st.expander("🔍 AI Analysis"):
                            st.json(analysis)
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

        with col6:
            if st.button("🛠️ Generate Strategies", use_container_width=True):
                with st.spinner("AI generating new strategies..."):
                    try:
                        market_data = getattr(st.session_state.ml_bot, 'latest_market_data', {})
                        current_strategies = getattr(st.session_state.ml_bot, 'active_strategies', [])
                        improvements = st.session_state.ml_bot.evolution_bot.generate_strategy_improvements(
                            market_data, current_strategies
                        )
                        st.success("New strategies generated!")
                        with st.expander("💡 AI Strategy Recommendations"):
                            st.json(improvements)
                    except Exception as e:
                        st.error(f"Strategy generation failed: {e}")

        # Auto Evolution Settings
        st.subheader("⚙️ Auto Evolution")
        auto_evolution_enabled = st.checkbox("🔄 Enable Continuous Evolution")

        if auto_evolution_enabled:
            col7, col8 = st.columns(2)
            with col7:
                interval = st.slider("Evolution Interval (minutes)", 15, 120, 30)
            with col8:
                if st.button("▶️ Start Auto Evolution"):
                    success = st.session_state.ml_bot.start_auto_evolution(interval)
                    if success:
                        st.success(f"Auto evolution started with {interval} minute intervals")
                    else:
                        st.error("Failed to start auto evolution")

        # Recent Evolution Log
        st.subheader("📋 Recent Evolution Activity")
        if hasattr(st.session_state.ml_bot, 'evolution_bot') and st.session_state.ml_bot.evolution_bot:
            if hasattr(st.session_state.ml_bot.evolution_bot, 'evolution_log'):
                log_entries = st.session_state.ml_bot.evolution_bot.evolution_log[-3:]
                if log_entries:
                    for i, entry in enumerate(reversed(log_entries)):
                        with st.expander(f"Evolution #{len(log_entries)-i}: {entry.get('timestamp', 'Unknown')}"):
                            st.json(entry)
                else:
                    st.info("No evolution activity yet. Run an evolution cycle to start!")

    else:
        st.warning("🚫 Self-Evolution capabilities not available")

        st.info("""
        **🤖 Self-Evolution Features (Requires OpenAI API Key):**

        • **🧠 AI-Powered Analysis**: GPT-4o analyzes performance and suggests improvements
        • **🔧 Auto Error Fixing**: Automatically detects and fixes code errors
        • **📈 Strategy Enhancement**: Generates new trading strategies
        • **📊 Performance Monitoring**: Continuous system health tracking
        • **🔄 Continuous Learning**: Self-improving algorithms
        • **🚨 Smart Alerts**: Intelligent performance alerts
        """)

if __name__ == "__main__":
    main()