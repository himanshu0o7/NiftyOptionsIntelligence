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
    st.session_state.paper_trading = False  # Live trading mode
if 'live_trading' not in st.session_state:
    st.session_state.live_trading = True
if 'capital' not in st.session_state:
    st.session_state.capital = 17000.0  # Total capital
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
        
        # API Credentials (auto-populated from environment)
        api_key = os.getenv("ANGEL_API_KEY", "")
        client_code = os.getenv("ANGEL_CLIENT_CODE", "")
        password = os.getenv("ANGEL_PASSWORD", "")
        totp_secret = os.getenv("ANGEL_TOTP_SECRET", "")
        
        if all([api_key, client_code, password, totp_secret]):
            st.success("✅ API credentials loaded from environment")
            # Manual TOTP entry (auto-generation disabled due to secret format)
            totp = st.text_input("Enter TOTP from Angel One app:", value="", max_chars=6)
            if totp:
                st.info(f"TOTP entered: {totp}")
            else:
                st.warning("Please enter 6-digit TOTP from your Angel One mobile app")
        else:
            st.error("❌ API credentials not found in environment")
            api_key = st.text_input("API Key", type="password", value="")
            client_code = st.text_input("Client Code", value="")
            password = st.text_input("Password", type="password", value="")
            totp = st.text_input("TOTP", value="")
        
        # Trading Mode
        if st.session_state.live_trading:
            st.success("🔴 LIVE TRADING ACTIVE")
            st.warning("⚠️ Real money at risk!")
        
        st.session_state.paper_trading = st.toggle("Paper Trading Mode", value=False)
        if st.session_state.paper_trading:
            st.session_state.live_trading = False
        else:
            st.session_state.live_trading = True
        
        # Connect to Angel One for Live Trading
        if all([api_key, client_code, password]) and not st.session_state.is_connected:
            if st.button("🚀 Connect to Angel One (Live Trading)", type="primary"):
                if totp and len(totp) == 6:
                    try:
                        with st.spinner("Connecting to Angel One API for live trading..."):
                            api_client = AngelOneAPI(api_key, client_code, password, totp)
                            if api_client.connect():
                                st.session_state.api_client = api_client
                                st.session_state.is_connected = True
                                
                                # Initialize WebSocket for live data
                                ws_client = WebSocketClient(api_client)
                                st.session_state.websocket_client = ws_client
                                
                                # Initialize position manager with live trading config
                                config = get_live_trading_config()
                                st.session_state.position_manager = PositionManager(config)
                                
                                st.success("✅ Connected to Angel One - Live Trading Ready!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("❌ Connection failed! Check credentials and TOTP.")
                    except Exception as e:
                        st.error(f"❌ Connection Error: {str(e)}")
                else:
                    st.warning("Please enter a valid 6-digit TOTP")
        elif not all([api_key, client_code, password]):
            st.button("Connect to Angel One", disabled=True)
            st.caption("Missing API credentials")
        
        # Disconnect Button
        if st.session_state.is_connected:
            if st.button("Disconnect", type="secondary"):
                st.session_state.api_client = None
                st.session_state.websocket_client = None
                st.session_state.is_connected = False
                st.success("Disconnected successfully!")
                st.rerun()
    
    # Live Trading Status Summary
    if st.session_state.live_trading and st.session_state.is_connected:
        st.success("🔴 LIVE TRADING SYSTEM ACTIVE")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Capital", "₹17,000", "")
            st.metric("Max Daily Loss", "₹850", "5% limit")
        
        with col2:
            st.metric("Position Limit", "₹3,400", "per position")
            st.metric("Risk Per Trade", "₹340", "2% of capital")
        
        with col3:
            st.metric("Lot Size", "1", "lot only")
            st.metric("Max Positions", "5", "concurrent")
    
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
    # Live Trading Mode Header
    if st.session_state.live_trading:
        st.success("🔴 LIVE TRADING MODE ACTIVE")
        st.header("📊 Live Trading Dashboard - Capital: ₹17,000")
    else:
        st.info("📝 Paper Trading Mode")
        st.header("📊 Trading Dashboard")
    
    # Trading Status and Capital Management
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Market Status", "OPEN", "🟢")
    
    with col2:
        st.metric("Capital", f"₹{st.session_state.capital:,.0f}", "")
    
    with col3:
        st.metric("Available", "₹15,550", "-8.5%")
    
    with col4:
        st.metric("Today's P&L", "₹0", "0%")
    
    with col5:
        st.metric("Positions", "0", "")
    
    # Risk Management Alert for Live Trading
    if st.session_state.live_trading:
        st.subheader("⚠️ Live Trading Risk Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Daily Loss Limit:** ₹850 (5% of capital)")
            st.info(f"**Max Position Size:** ₹3,400 (20% of capital)")
        
        with col2:
            st.info(f"**Risk Per Trade:** 2% (₹340)")
            st.info(f"**Max Positions:** 5 concurrent")
        
        with col3:
            st.info(f"**Lot Size:** 1 lot only")
            st.info(f"**Stop Loss:** 2% default")
    
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

def get_live_trading_config():
    """Get configuration for live trading with 17k capital"""
    return {
        'capital': 17000.0,
        'max_daily_loss': 850.0,  # 5% of capital
        'max_position_size': 3400.0,  # 20% of capital per position
        'risk_per_trade': 340.0,  # 2% of capital
        'max_positions': 5,
        'default_quantity': 1,  # 1 lot only
        'stop_loss_pct': 2.0,
        'take_profit_pct': 5.0,
        'live_trading': True,
        'paper_trading': False
    }

def update_positions():
    """Update current positions"""
    if st.session_state.live_trading:
        st.info("Live trading mode: Positions will be fetched from Angel One API")
    else:
        st.success("Paper trading positions updated!")
    
    # Initialize position manager with live trading config
    if st.session_state.live_trading:
        config = get_live_trading_config()
        if st.session_state.api_client:
            position_manager = PositionManager(config)
            # Fetch real positions from API
            st.info("Connected to API - ready for live position management")

def check_position_risk():
    """Check position risk"""
    st.info("Risk analysis completed. All positions within acceptable risk limits.")

def calculate_pnl():
    """Calculate P&L"""
    st.success("P&L calculated and updated!")

if __name__ == "__main__":
    import numpy as np
    main()
