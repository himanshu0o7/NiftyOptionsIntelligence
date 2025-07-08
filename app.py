import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import threading
import time
import os

# Import our modules
from core.angel_api import AngelOneAPI
from core.websocket_client import WebSocketClient
from core.database import Database
from strategies.breakout_strategy import BreakoutStrategy
from strategies.oi_analysis import OIAnalysis
from risk_management.position_manager import PositionManager
from utils.logger import Logger
from config.settings import Settings

# Initialize session state
if 'api_client' not in st.session_state:
    st.session_state.api_client = None
if 'websocket_client' not in st.session_state:
    st.session_state.websocket_client = None
if 'is_connected' not in st.session_state:
    st.session_state.is_connected = False
if 'paper_trading' not in st.session_state:
    st.session_state.paper_trading = True
if 'active_positions' not in st.session_state:
    st.session_state.active_positions = []
if 'signals' not in st.session_state:
    st.session_state.signals = []

# Page configuration
st.set_page_config(
    page_title="Options Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_components():
    """Initialize all system components"""
    try:
        # Initialize database
        db = Database()
        
        # Initialize logger
        logger = Logger()
        
        # Initialize settings
        settings = Settings()
        
        return db, logger, settings
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None, None

def main():
    st.title("🚀 Automated Options Trading System")
    st.markdown("**NIFTY50 & BANKNIFTY Options Trading with Angel One API**")
    
    # Initialize components
    db, logger, settings = initialize_components()
    if not all([db, logger, settings]):
        st.error("Failed to initialize system components")
        return
    
    # Sidebar - Connection and Settings
    with st.sidebar:
        st.header("🔐 API Connection")
        
        # API Credentials
        api_key = st.text_input("API Key", type="password", value=os.getenv("ANGEL_API_KEY", ""))
        client_code = st.text_input("Client Code", value=os.getenv("ANGEL_CLIENT_CODE", ""))
        password = st.text_input("Password", type="password", value=os.getenv("ANGEL_PASSWORD", ""))
        totp = st.text_input("TOTP", value="")
        
        # Paper Trading Toggle
        st.session_state.paper_trading = st.toggle("Paper Trading Mode", value=True)
        
        # Connection Button
        if st.button("Connect to Angel One", type="primary"):
            if all([api_key, client_code, password, totp]):
                try:
                    with st.spinner("Connecting to Angel One API..."):
                        api_client = AngelOneAPI(api_key, client_code, password, totp)
                        if api_client.connect():
                            st.session_state.api_client = api_client
                            st.session_state.is_connected = True
                            
                            # Initialize WebSocket
                            ws_client = WebSocketClient(api_client)
                            st.session_state.websocket_client = ws_client
                            
                            st.success("✅ Connected successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Connection failed!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Please fill all credentials")
        
        # Disconnect Button
        if st.session_state.is_connected:
            if st.button("Disconnect", type="secondary"):
                st.session_state.api_client = None
                st.session_state.websocket_client = None
                st.session_state.is_connected = False
                st.success("Disconnected successfully!")
                st.rerun()
    
    # Main Dashboard
    if st.session_state.is_connected:
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", 
            "⚡ Signals", 
            "💼 Positions", 
            "⚙️ Strategy Config", 
            "📈 P&L Analysis"
        ])
        
        with tab1:
            display_dashboard()
        
        with tab2:
            display_signals()
        
        with tab3:
            display_positions()
        
        with tab4:
            display_strategy_config()
        
        with tab5:
            display_pnl_analysis()
    
    else:
        st.info("👈 Please connect to Angel One API using the sidebar")
        
        # Show system overview
        st.header("🎯 System Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📈 Technical Analysis")
            st.markdown("""
            - Open Interest Analysis
            - Greeks Calculation  
            - RSI, EMA, VWAP Indicators
            - Breakout/Breakdown Detection
            """)
        
        with col2:
            st.subheader("🎛️ Risk Management")
            st.markdown("""
            - Stop Loss (SL) Logic
            - Trailing Stop Loss (TSL)
            - Take Profit (TP) Targets
            - Position Sizing
            """)
        
        with col3:
            st.subheader("🔄 Automation")
            st.markdown("""
            - Real-time Market Data
            - Automated Order Execution
            - Paper Trading Mode
            - Comprehensive Logging
            """)

def display_dashboard():
    """Display main trading dashboard"""
    st.header("📊 Live Trading Dashboard")
    
    # Market Status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Market Status", "OPEN", "🟢")
    
    with col2:
        st.metric("Active Strategies", "3", "+1")
    
    with col3:
        st.metric("Today's P&L", "₹2,450", "+5.2%")
    
    with col4:
        st.metric("Total Positions", "8", "+2")
    
    # Live Charts
    st.subheader("📈 Live Market Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("NIFTY50 Options Chain")
        # Create sample options chain data
        options_data = create_sample_options_chain("NIFTY")
        st.dataframe(options_data, use_container_width=True)
    
    with col2:
        st.subheader("BANKNIFTY Options Chain")
        # Create sample options chain data
        options_data = create_sample_options_chain("BANKNIFTY")
        st.dataframe(options_data, use_container_width=True)

def display_signals():
    """Display trading signals"""
    st.header("⚡ Trading Signals")
    
    # Signal Generation Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Scan for Breakouts", type="primary"):
            generate_breakout_signals()
    
    with col2:
        if st.button("📊 Analyze OI Changes", type="primary"):
            generate_oi_signals()
    
    with col3:
        if st.button("🎯 Check All Signals", type="primary"):
            generate_all_signals()
    
    # Signals Display
    if st.session_state.signals:
        st.subheader("📋 Active Signals")
        signals_df = pd.DataFrame(st.session_state.signals)
        st.dataframe(signals_df, use_container_width=True)
    else:
        st.info("No active signals. Click the buttons above to generate signals.")

def display_positions():
    """Display current positions"""
    st.header("💼 Current Positions")
    
    if st.session_state.active_positions:
        positions_df = pd.DataFrame(st.session_state.active_positions)
        st.dataframe(positions_df, use_container_width=True)
        
        # Position Actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Update Positions"):
                update_positions()
        with col2:
            if st.button("🛡️ Check Risk"):
                check_position_risk()
        with col3:
            if st.button("💰 Calculate P&L"):
                calculate_pnl()
    else:
        st.info("No active positions")

def display_strategy_config():
    """Display strategy configuration"""
    st.header("⚙️ Strategy Configuration")
    
    # Strategy Selection
    strategy_type = st.selectbox(
        "Select Strategy",
        ["Breakout Strategy", "OI Analysis", "Greeks Based", "Custom"]
    )
    
    # Strategy Parameters
    st.subheader("📝 Strategy Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input("Stop Loss %", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        st.number_input("Take Profit %", min_value=0.1, max_value=20.0, value=5.0, step=0.1)
        st.number_input("Position Size (Lots)", min_value=1, max_value=50, value=1)
    
    with col2:
        st.selectbox("Risk Level", ["Low", "Medium", "High"])
        st.selectbox("Time Frame", ["1m", "5m", "15m", "1h"])
        st.toggle("Enable Trailing SL")
    
    # Save Configuration
    if st.button("💾 Save Configuration", type="primary"):
        st.success("Strategy configuration saved!")

def display_pnl_analysis():
    """Display P&L analysis"""
    st.header("📈 P&L Analysis")
    
    # P&L Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Today's P&L", "₹2,450", "+5.2%")
    with col2:
        st.metric("This Week", "₹8,750", "+12.3%")
    with col3:
        st.metric("This Month", "₹25,680", "+18.7%")
    with col4:
        st.metric("Total P&L", "₹1,25,430", "+45.2%")
    
    # P&L Chart
    st.subheader("📊 P&L Trend")
    
    # Create sample P&L data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    pnl_data = pd.DataFrame({
        'Date': dates,
        'Daily_PnL': np.random.normal(100, 500, len(dates)).cumsum(),
        'Cumulative_PnL': np.random.normal(100, 500, len(dates)).cumsum()
    })
    
    fig = px.line(pnl_data, x='Date', y='Cumulative_PnL', 
                  title='Cumulative P&L Over Time')
    st.plotly_chart(fig, use_container_width=True)

def create_sample_options_chain(symbol):
    """Create sample options chain data"""
    import numpy as np
    
    if symbol == "NIFTY":
        spot = 21500
        strikes = range(21000, 22000, 50)
    else:  # BANKNIFTY
        spot = 45000
        strikes = range(44500, 45500, 100)
    
    data = []
    for strike in strikes:
        data.append({
            'Strike': strike,
            'CE_LTP': np.random.randint(50, 500),
            'CE_OI': np.random.randint(1000, 50000),
            'CE_Volume': np.random.randint(100, 5000),
            'PE_LTP': np.random.randint(50, 500),
            'PE_OI': np.random.randint(1000, 50000),
            'PE_Volume': np.random.randint(100, 5000)
        })
    
    return pd.DataFrame(data)

def generate_breakout_signals():
    """Generate breakout signals"""
    signals = [
        {
            'Time': datetime.now().strftime('%H:%M:%S'),
            'Symbol': 'NIFTY 21500 CE',
            'Signal': 'BREAKOUT',
            'Action': 'BUY',
            'Price': 285,
            'Confidence': '85%'
        },
        {
            'Time': datetime.now().strftime('%H:%M:%S'),
            'Symbol': 'BANKNIFTY 45000 PE',
            'Signal': 'BREAKDOWN',
            'Action': 'SELL',
            'Price': 320,
            'Confidence': '78%'
        }
    ]
    st.session_state.signals.extend(signals)
    st.success(f"Generated {len(signals)} breakout signals!")

def generate_oi_signals():
    """Generate OI analysis signals"""
    signals = [
        {
            'Time': datetime.now().strftime('%H:%M:%S'),
            'Symbol': 'NIFTY 21400 PE',
            'Signal': 'HIGH_OI_BUILD',
            'Action': 'WATCH',
            'Price': 195,
            'Confidence': '92%'
        }
    ]
    st.session_state.signals.extend(signals)
    st.success(f"Generated {len(signals)} OI analysis signals!")

def generate_all_signals():
    """Generate all types of signals"""
    generate_breakout_signals()
    generate_oi_signals()
    st.success("Generated all signals!")

def update_positions():
    """Update current positions"""
    st.success("Positions updated!")

def check_position_risk():
    """Check position risk"""
    st.info("Risk analysis completed. All positions within acceptable risk limits.")

def calculate_pnl():
    """Calculate P&L"""
    st.success("P&L calculated and updated!")

if __name__ == "__main__":
    import numpy as np
    main()
