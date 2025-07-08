import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from data.instruments import instrument_manager
from utils.helpers import helper
from indicators.technical_indicators import TechnicalIndicators

def show_trading_dashboard():
    """Display the main trading dashboard"""
    
    st.header("📊 Live Trading Dashboard")
    
    # Market Status Header
    show_market_status()
    
    # Key Metrics Row
    show_key_metrics()
    
    # Main Content Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Live Charts", 
        "🎯 Options Chain", 
        "📊 Technical Analysis", 
        "💹 Market Depth"
    ])
    
    with tab1:
        show_live_charts()
    
    with tab2:
        show_options_chain()
    
    with tab3:
        show_technical_analysis()
    
    with tab4:
        show_market_depth()

def show_market_status():
    """Display current market status"""
    market_status = helper.get_market_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🟢" if market_status.get('is_open', False) else "🔴"
        status_text = "OPEN" if market_status.get('is_open', False) else "CLOSED"
        st.metric("Market Status", f"{status_color} {status_text}")
    
    with col2:
        current_time = market_status.get('current_time', datetime.now())
        st.metric("Current Time", current_time.strftime("%H:%M:%S IST"))
    
    with col3:
        next_event = market_status.get('next_event', 'Unknown')
        st.metric("Next Event", next_event)
    
    with col4:
        if 'time_to_next_event' in market_status:
            time_delta = market_status['time_to_next_event']
            hours, remainder = divmod(time_delta.total_seconds(), 3600)
            minutes, _ = divmod(remainder, 60)
            st.metric("Time Remaining", f"{int(hours):02d}:{int(minutes):02d}")

def show_key_metrics():
    """Display key trading metrics"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Sample data - in real implementation, fetch from API
        st.metric("NIFTY", "21,547", "+127 (+0.59%)", delta_color="normal")
    
    with col2:
        st.metric("BANKNIFTY", "45,123", "-89 (-0.20%)", delta_color="normal")
    
    with col3:
        st.metric("Active Positions", "8", "+2")
    
    with col4:
        st.metric("Today's P&L", "₹2,450", "+₹340")
    
    with col5:
        st.metric("Portfolio Value", "₹1,25,000", "+2.1%")

def show_live_charts():
    """Display live price charts"""
    st.subheader("📈 Real-time Price Charts")
    
    # Chart selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_underlying = st.selectbox(
            "Select Underlying",
            ["NIFTY", "BANKNIFTY"]
        )
        
        timeframe = st.selectbox(
            "Timeframe",
            ["1m", "5m", "15m", "1h", "1d"]
        )
        
        chart_type = st.selectbox(
            "Chart Type",
            ["Candlestick", "Line", "Area"]
        )
    
    with col2:
        # Generate sample chart data
        chart_data = generate_sample_chart_data(selected_underlying, timeframe)
        
        if chart_type == "Candlestick":
            fig = create_candlestick_chart(chart_data, selected_underlying)
        elif chart_type == "Line":
            fig = create_line_chart(chart_data, selected_underlying)
        else:  # Area
            fig = create_area_chart(chart_data, selected_underlying)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicators panel
    st.subheader("📊 Technical Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("RSI (14)", "67.8", "+2.3")
        st.metric("MACD", "12.45", "+0.78")
    
    with col2:
        st.metric("EMA (9)", "21,534", "+45")
        st.metric("EMA (21)", "21,489", "+23")
    
    with col3:
        st.metric("Support", "21,450", "")
        st.metric("Resistance", "21,650", "")

def show_options_chain():
    """Display options chain data"""
    st.subheader("🎯 Options Chain Analysis")
    
    # Options chain controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        underlying = st.selectbox("Underlying", ["NIFTY", "BANKNIFTY"], key="oc_underlying")
    
    with col2:
        # Get available expiries
        expiries = get_available_expiries(underlying)
        selected_expiry = st.selectbox("Expiry", expiries, key="oc_expiry")
    
    with col3:
        spot_price = st.number_input("Spot Price", value=21547.0 if underlying == "NIFTY" else 45123.0)
    
    # Display options chain
    if selected_expiry:
        options_chain_data = get_options_chain_data(underlying, selected_expiry, spot_price)
        
        if not options_chain_data.empty:
            # Style the options chain
            styled_chain = style_options_chain(options_chain_data, spot_price)
            st.dataframe(styled_chain, use_container_width=True, height=600)
            
            # Options chain analytics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Put-Call Ratio")
                pcr_data = calculate_pcr(options_chain_data)
                st.metric("PCR (OI)", f"{pcr_data['pcr_oi']:.2f}")
                st.metric("PCR (Volume)", f"{pcr_data['pcr_volume']:.2f}")
            
            with col2:
                st.subheader("🎯 Max Pain Analysis")
                max_pain = calculate_max_pain(options_chain_data)
                st.metric("Max Pain Level", f"{max_pain:,.0f}")
                deviation = abs(spot_price - max_pain) / spot_price * 100
                st.metric("Deviation from Max Pain", f"{deviation:.1f}%")

def show_technical_analysis():
    """Display technical analysis"""
    st.subheader("📊 Technical Analysis")
    
    # Technical analysis controls
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], key="ta_symbol")
        
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type", 
            ["Trend Analysis", "Momentum", "Volatility", "Support/Resistance"]
        )
    
    # Generate technical analysis
    technical_data = generate_technical_analysis(symbol, analysis_type)
    
    if analysis_type == "Trend Analysis":
        show_trend_analysis(technical_data)
    elif analysis_type == "Momentum":
        show_momentum_analysis(technical_data)
    elif analysis_type == "Volatility":
        show_volatility_analysis(technical_data)
    else:
        show_support_resistance_analysis(technical_data)

def show_market_depth():
    """Display market depth and order book"""
    st.subheader("💹 Market Depth")
    
    # Market depth controls
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], key="md_symbol")
    
    with col2:
        contract_type = st.selectbox("Contract", ["Futures", "Options"], key="md_contract")
    
    # Display order book
    if contract_type == "Futures":
        order_book = generate_futures_order_book(symbol)
    else:
        # For options, show ATM strikes
        order_book = generate_options_order_book(symbol)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Bids")
        st.dataframe(order_book['bids'], use_container_width=True)
    
    with col2:
        st.subheader("🔴 Asks")
        st.dataframe(order_book['asks'], use_container_width=True)
    
    # Volume analysis
    st.subheader("📊 Volume Analysis")
    volume_chart = create_volume_chart(symbol)
    st.plotly_chart(volume_chart, use_container_width=True)

# Helper functions for dashboard

def generate_sample_chart_data(underlying: str, timeframe: str) -> pd.DataFrame:
    """Generate sample chart data"""
    # In real implementation, fetch from API
    periods = 100
    base_price = 21547 if underlying == "NIFTY" else 45123
    
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
    
    # Generate realistic price movements
    returns = np.random.normal(0, 0.002, periods)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, periods)
    })
    
    return df

def create_candlestick_chart(data: pd.DataFrame, title: str) -> go.Figure:
    """Create candlestick chart"""
    fig = go.Figure(data=go.Candlestick(
        x=data['timestamp'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name=title
    ))
    
    fig.update_layout(
        title=f"{title} - Candlestick Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        height=500
    )
    
    return fig

def create_line_chart(data: pd.DataFrame, title: str) -> go.Figure:
    """Create line chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['close'],
        mode='lines',
        name=title,
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=f"{title} - Price Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        height=500
    )
    
    return fig

def create_area_chart(data: pd.DataFrame, title: str) -> go.Figure:
    """Create area chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['close'],
        fill='tonexty',
        mode='lines',
        name=title,
        fillcolor='rgba(0, 100, 80, 0.2)',
        line=dict(color='rgba(0, 100, 80, 1)')
    ))
    
    fig.update_layout(
        title=f"{title} - Area Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        height=500
    )
    
    return fig

def get_available_expiries(underlying: str) -> list:
    """Get available expiry dates"""
    # In real implementation, fetch from instrument manager
    today = datetime.now()
    expiries = []
    
    # Generate next 4 weekly expiries
    for i in range(4):
        if underlying == "NIFTY":
            # NIFTY expires on Thursday
            days_ahead = (3 - today.weekday()) % 7 + (i * 7)
        else:
            # BANKNIFTY expires on Wednesday  
            days_ahead = (2 - today.weekday()) % 7 + (i * 7)
        
        if days_ahead == 0:
            days_ahead = 7
            
        expiry_date = today + timedelta(days=days_ahead)
        expiries.append(expiry_date.strftime("%d-%b-%Y"))
    
    return expiries

def get_options_chain_data(underlying: str, expiry: str, spot_price: float) -> pd.DataFrame:
    """Get options chain data"""
    # Generate sample options chain
    if underlying == "NIFTY":
        strike_step = 50
        base_strike = round(spot_price / strike_step) * strike_step
    else:
        strike_step = 100
        base_strike = round(spot_price / strike_step) * strike_step
    
    strikes = []
    for i in range(-10, 11):
        strikes.append(base_strike + (i * strike_step))
    
    options_data = []
    for strike in strikes:
        # Generate sample data
        call_oi = np.random.randint(100, 5000)
        put_oi = np.random.randint(100, 5000)
        call_volume = np.random.randint(10, 1000)
        put_volume = np.random.randint(10, 1000)
        
        # Calculate LTP based on moneyness
        call_ltp = max(0.05, max(0, spot_price - strike) + np.random.uniform(5, 50))
        put_ltp = max(0.05, max(0, strike - spot_price) + np.random.uniform(5, 50))
        
        options_data.append({
            'Strike': strike,
            'Call_OI': call_oi,
            'Call_Volume': call_volume,
            'Call_LTP': round(call_ltp, 2),
            'Put_LTP': round(put_ltp, 2),
            'Put_Volume': put_volume,
            'Put_OI': put_oi
        })
    
    return pd.DataFrame(options_data)

def style_options_chain(df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
    """Style options chain for display"""
    # Add styling information (in real Streamlit, you'd use st.dataframe with styling)
    df_styled = df.copy()
    
    # Mark ATM strike
    atm_strike = df_styled.loc[(df_styled['Strike'] - spot_price).abs().idxmin(), 'Strike']
    
    return df_styled

def calculate_pcr(options_data: pd.DataFrame) -> dict:
    """Calculate Put-Call Ratio"""
    total_call_oi = options_data['Call_OI'].sum()
    total_put_oi = options_data['Put_OI'].sum()
    total_call_volume = options_data['Call_Volume'].sum()
    total_put_volume = options_data['Put_Volume'].sum()
    
    pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
    pcr_volume = total_put_volume / total_call_volume if total_call_volume > 0 else 0
    
    return {
        'pcr_oi': pcr_oi,
        'pcr_volume': pcr_volume
    }

def calculate_max_pain(options_data: pd.DataFrame) -> float:
    """Calculate max pain level"""
    max_pain_values = {}
    
    for _, row in options_data.iterrows():
        strike = row['Strike']
        call_oi = row['Call_OI']
        put_oi = row['Put_OI']
        
        # Calculate pain for different price levels
        for price in options_data['Strike']:
            if price not in max_pain_values:
                max_pain_values[price] = 0
            
            # Add call pain
            if price > strike:
                max_pain_values[price] += (price - strike) * call_oi
            
            # Add put pain
            if price < strike:
                max_pain_values[price] += (strike - price) * put_oi
    
    # Find price with minimum pain
    if max_pain_values:
        max_pain = min(max_pain_values, key=max_pain_values.get)
        return max_pain
    
    return 0

def generate_technical_analysis(symbol: str, analysis_type: str) -> dict:
    """Generate technical analysis data"""
    # Sample technical analysis data
    return {
        'symbol': symbol,
        'analysis_type': analysis_type,
        'data': {
            'rsi': 67.8,
            'macd': 12.45,
            'ema_9': 21534,
            'ema_21': 21489,
            'support': 21450,
            'resistance': 21650,
            'trend': 'Bullish',
            'momentum': 'Strong'
        }
    }

def show_trend_analysis(data: dict):
    """Show trend analysis"""
    st.write("**Trend Direction:** Bullish")
    st.write("**EMA Crossover:** 9 EMA above 21 EMA")
    st.write("**Price Action:** Higher highs and higher lows")

def show_momentum_analysis(data: dict):
    """Show momentum analysis"""
    st.write("**RSI:** 67.8 (Bullish)")
    st.write("**MACD:** 12.45 (Positive)")
    st.write("**Momentum:** Strong upward momentum")

def show_volatility_analysis(data: dict):
    """Show volatility analysis"""
    st.write("**Implied Volatility:** 18.5%")
    st.write("**Historical Volatility:** 16.2%")
    st.write("**Volatility Trend:** Increasing")

def show_support_resistance_analysis(data: dict):
    """Show support/resistance analysis"""
    st.write("**Immediate Support:** 21,450")
    st.write("**Immediate Resistance:** 21,650")
    st.write("**Key Levels:** 21,200 (Strong Support), 21,800 (Strong Resistance)")

def generate_futures_order_book(symbol: str) -> dict:
    """Generate sample futures order book"""
    bids = pd.DataFrame({
        'Price': [21545, 21544, 21543, 21542, 21541],
        'Quantity': [150, 200, 300, 175, 225],
        'Orders': [5, 8, 12, 7, 9]
    })
    
    asks = pd.DataFrame({
        'Price': [21546, 21547, 21548, 21549, 21550],
        'Quantity': [175, 250, 180, 300, 200],
        'Orders': [6, 10, 8, 15, 7]
    })
    
    return {'bids': bids, 'asks': asks}

def generate_options_order_book(symbol: str) -> dict:
    """Generate sample options order book"""
    # Similar to futures but for options
    return generate_futures_order_book(symbol)

def create_volume_chart(symbol: str) -> go.Figure:
    """Create volume analysis chart"""
    # Generate sample volume data
    periods = 20
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='H')
    volumes = np.random.randint(1000, 10000, periods)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=dates,
        y=volumes,
        name="Volume",
        marker_color='rgba(55, 83, 109, 0.7)'
    ))
    
    fig.update_layout(
        title=f"{symbol} - Volume Analysis",
        xaxis_title="Time",
        yaxis_title="Volume",
        height=300
    )
    
    return fig
