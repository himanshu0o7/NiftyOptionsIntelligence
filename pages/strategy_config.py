import streamlit as st
import json
import pandas as pd
import sys
from pathlib import Path

from telegram_alerts import send_telegram_alert

st.set_page_config(page_title="Strategy Configuration", layout="wide")
st.write("✅ App Loaded")


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from datetime import datetime
from typing import Dict, List
from strategies.breakout_strategy import BreakoutStrategy
from strategies.oi_analysis import OIAnalysis
from config.settings import Settings

MODULE_NAME = "strategy_config"

def show_strategy_config():
    """Display strategy configuration page"""

    try:
        st.header("⚙️ Strategy Configuration")

        # Strategy selection and management
        strategy_tab1, strategy_tab2, strategy_tab3 = st.tabs([
            "🎯 Active Strategies",
            "⚙️ Configure Strategy",
            "📊 Backtest Results"
        ])

        with strategy_tab1:
            show_active_strategies()

        with strategy_tab2:
            show_strategy_configuration()

        with strategy_tab3:
            show_backtest_results()
    except Exception as exc:
        send_telegram_alert(f"{MODULE_NAME} error: {exc}")
        st.error("An error occurred while loading the Strategy Configuration page.")

def show_active_strategies():
    """Display and manage active strategies"""
    st.subheader("🎯 Active Trading Strategies")
    
    # Strategy status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Strategies", "3", "+1")
    
    with col2:
        st.metric("Active", "2", "")
    
    with col3:
        st.metric("Paused", "1", "")
    
    with col4:
        st.metric("Avg Performance", "+15.2%", "+2.1%")
    
    st.divider()
    
    # Active strategies table
    strategies_data = get_active_strategies_data()
    
    if not strategies_data.empty:
        # Add action buttons
        for idx, row in strategies_data.iterrows():
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
                
                with col1:
                    status_icon = "🟢" if row['Status'] == 'Active' else "🟡"
                    st.write(f"**{status_icon} {row['Strategy']}**")
                    st.caption(f"Symbols: {row['Symbols']}")
                
                with col2:
                    st.metric("P&L", row['PnL'], row['PnL_Change'])
                
                with col3:
                    st.metric("Trades", str(row['Trades']), row['Win_Rate'])
                
                with col4:
                    if st.button("⏸️" if row['Status'] == 'Active' else "▶️", 
                               key=f"toggle_{idx}"):
                        toggle_strategy_status(row['Strategy'])
                
                with col5:
                    if st.button("⚙️", key=f"config_{idx}"):
                        st.session_state.config_strategy = row['Strategy']
                
                st.divider()
    else:
        st.info("No active strategies. Configure a new strategy to get started.")

def show_strategy_configuration():
    """Show strategy configuration interface"""
    st.subheader("⚙️ Strategy Configuration")
    
    # Strategy type selection
    strategy_type = st.selectbox(
        "Select Strategy Type",
        ["Breakout Strategy", "OI Analysis Strategy", "Greeks Based Strategy", "Custom Strategy"]
    )
    
    if strategy_type == "Breakout Strategy":
        configure_breakout_strategy()
    elif strategy_type == "OI Analysis Strategy":
        configure_oi_analysis_strategy()
    elif strategy_type == "Greeks Based Strategy":
        configure_greeks_strategy()
    else:
        configure_custom_strategy()

def configure_breakout_strategy():
    """Configure breakout strategy parameters"""
    st.subheader("📈 Breakout Strategy Configuration")
    
    with st.form("breakout_strategy_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Parameters**")
            
            strategy_name = st.text_input("Strategy Name", value="Breakout_NIFTY")
            
            symbols = st.multiselect(
                "Select Symbols",
                ["NIFTY", "BANKNIFTY", "FINNIFTY"],
                default=["NIFTY"]
            )
            
            timeframe = st.selectbox(
                "Timeframe",
                ["1m", "5m", "15m", "30m", "1h"],
                index=2
            )
            
            breakout_threshold = st.slider(
                "Breakout Threshold (%)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1
            )
            
            volume_multiplier = st.slider(
                "Volume Multiplier",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.1
            )
        
        with col2:
            st.write("**Risk Management**")
            
            position_size = st.number_input(
                "Position Size (Lots)",
                min_value=1,
                max_value=50,
                value=1
            )
            
            stop_loss = st.slider(
                "Stop Loss (%)",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.1
            )
            
            take_profit = st.slider(
                "Take Profit (%)",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=0.1
            )
            
            trailing_stop = st.checkbox("Enable Trailing Stop")
            
            if trailing_stop:
                trailing_pct = st.slider(
                    "Trailing Stop (%)",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.0,
                    step=0.1
                )
            else:
                trailing_pct = 0
            
            max_positions = st.number_input(
                "Max Concurrent Positions",
                min_value=1,
                max_value=10,
                value=3
            )
        
        st.write("**Advanced Settings**")
        
        col3, col4 = st.columns(2)
        
        with col3:
            confirmation_candles = st.number_input(
                "Confirmation Candles",
                min_value=1,
                max_value=5,
                value=2
            )
            
            lookback_period = st.number_input(
                "Lookback Period",
                min_value=10,
                max_value=50,
                value=20
            )
        
        with col4:
            market_hours_only = st.checkbox("Trade Only During Market Hours", value=True)
            
            paper_trading = st.checkbox("Paper Trading Mode", value=True)
        
        submitted = st.form_submit_button("💾 Save Strategy Configuration")
        
        if submitted:
            config = {
                "strategy_name": strategy_name,
                "strategy_type": "breakout",
                "symbols": symbols,
                "timeframe": timeframe,
                "breakout_threshold": breakout_threshold,
                "volume_multiplier": volume_multiplier,
                "position_size": position_size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop": trailing_pct if trailing_stop else None,
                "max_positions": max_positions,
                "confirmation_candles": confirmation_candles,
                "lookback_period": lookback_period,
                "market_hours_only": market_hours_only,
                "paper_trading": paper_trading,
                "created_at": datetime.now().isoformat()
            }
            
            save_strategy_config(config)
            st.success(f"✅ Strategy '{strategy_name}' configured successfully!")

def configure_oi_analysis_strategy():
    """Configure OI analysis strategy parameters"""
    st.subheader("📊 Open Interest Analysis Strategy")
    
    with st.form("oi_analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Parameters**")
            
            strategy_name = st.text_input("Strategy Name", value="OI_Analysis_BANKNIFTY")
            
            symbols = st.multiselect(
                "Select Symbols",
                ["NIFTY", "BANKNIFTY"],
                default=["BANKNIFTY"]
            )
            
            oi_change_threshold = st.slider(
                "OI Change Threshold (%)",
                min_value=10.0,
                max_value=50.0,
                value=20.0,
                step=5.0
            )
            
            volume_oi_ratio = st.slider(
                "Volume/OI Ratio Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05
            )
            
            max_pain_deviation = st.slider(
                "Max Pain Deviation (%)",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.1
            )
        
        with col2:
            st.write("**PCR Analysis**")
            
            pcr_low_threshold = st.slider(
                "PCR Low Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.7,
                step=0.05
            )
            
            pcr_high_threshold = st.slider(
                "PCR High Threshold", 
                min_value=1.0,
                max_value=2.0,
                value=1.3,
                step=0.05
            )
            
            st.write("**Risk Management**")
            
            position_size = st.number_input(
                "Position Size (Lots)",
                min_value=1,
                max_value=20,
                value=1
            )
            
            stop_loss = st.slider(
                "Stop Loss (%)",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5
            )
            
            take_profit = st.slider(
                "Take Profit (%)",
                min_value=2.0,
                max_value=15.0,
                value=6.0,
                step=0.5
            )
        
        st.write("**Analysis Settings**")
        
        col3, col4 = st.columns(2)
        
        with col3:
            analyze_call_writing = st.checkbox("Analyze Call Writing", value=True)
            analyze_put_writing = st.checkbox("Analyze Put Writing", value=True)
            
        with col4:
            oi_buildup_analysis = st.checkbox("OI Buildup Analysis", value=True)
            max_pain_analysis = st.checkbox("Max Pain Analysis", value=True)
        
        paper_trading = st.checkbox("Paper Trading Mode", value=True)
        
        submitted = st.form_submit_button("💾 Save OI Strategy Configuration")
        
        if submitted:
            config = {
                "strategy_name": strategy_name,
                "strategy_type": "oi_analysis",
                "symbols": symbols,
                "oi_change_threshold": oi_change_threshold,
                "volume_oi_ratio_threshold": volume_oi_ratio,
                "max_pain_deviation": max_pain_deviation,
                "pcr_threshold": {
                    "low": pcr_low_threshold,
                    "high": pcr_high_threshold
                },
                "position_size": position_size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "analyze_call_writing": analyze_call_writing,
                "analyze_put_writing": analyze_put_writing,
                "oi_buildup_analysis": oi_buildup_analysis,
                "max_pain_analysis": max_pain_analysis,
                "paper_trading": paper_trading,
                "created_at": datetime.now().isoformat()
            }
            
            save_strategy_config(config)
            st.success(f"✅ OI Analysis Strategy '{strategy_name}' configured successfully!")

def configure_greeks_strategy():
    """Configure Greeks-based strategy"""
    st.subheader("🏛️ Greeks-Based Strategy Configuration")
    
    with st.form("greeks_strategy_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Parameters**")
            
            strategy_name = st.text_input("Strategy Name", value="Greeks_Delta_Neutral")
            
            symbols = st.multiselect(
                "Select Symbols",
                ["NIFTY", "BANKNIFTY"],
                default=["NIFTY"]
            )
            
            st.write("**Delta Management**")
            
            target_delta = st.slider(
                "Target Portfolio Delta",
                min_value=-100.0,
                max_value=100.0,
                value=0.0,
                step=5.0
            )
            
            delta_tolerance = st.slider(
                "Delta Tolerance",
                min_value=5.0,
                max_value=50.0,
                value=20.0,
                step=5.0
            )
            
            rebalance_frequency = st.selectbox(
                "Rebalance Frequency",
                ["Real-time", "Hourly", "Daily"],
                index=1
            )
        
        with col2:
            st.write("**Greeks Thresholds**")
            
            gamma_threshold = st.slider(
                "Gamma Threshold",
                min_value=0.01,
                max_value=0.1,
                value=0.05,
                step=0.01
            )
            
            theta_threshold = st.slider(
                "Theta Threshold",
                min_value=-100.0,
                max_value=-10.0,
                value=-50.0,
                step=5.0
            )
            
            vega_threshold = st.slider(
                "Vega Threshold",
                min_value=5.0,
                max_value=50.0,
                value=20.0,
                step=2.5
            )
            
            iv_threshold = st.slider(
                "IV Threshold (%)",
                min_value=15.0,
                max_value=35.0,
                value=25.0,
                step=1.0
            )
        
        st.write("**Position Management**")
        
        col3, col4 = st.columns(2)
        
        with col3:
            max_positions = st.number_input(
                "Max Positions",
                min_value=2,
                max_value=20,
                value=8
            )
            
            position_size = st.number_input(
                "Position Size (Lots)",
                min_value=1,
                max_value=10,
                value=2
            )
        
        with col4:
            stop_loss = st.slider(
                "Stop Loss (%)",
                min_value=5.0,
                max_value=20.0,
                value=10.0,
                step=1.0
            )
            
            profit_target = st.slider(
                "Profit Target (%)",
                min_value=5.0,
                max_value=30.0,
                value=15.0,
                step=1.0
            )
        
        hedge_enabled = st.checkbox("Enable Dynamic Hedging", value=True)
        paper_trading = st.checkbox("Paper Trading Mode", value=True)
        
        submitted = st.form_submit_button("💾 Save Greeks Strategy Configuration")
        
        if submitted:
            config = {
                "strategy_name": strategy_name,
                "strategy_type": "greeks_based",
                "symbols": symbols,
                "target_delta": target_delta,
                "delta_tolerance": delta_tolerance,
                "rebalance_frequency": rebalance_frequency,
                "gamma_threshold": gamma_threshold,
                "theta_threshold": theta_threshold,
                "vega_threshold": vega_threshold,
                "iv_threshold": iv_threshold,
                "max_positions": max_positions,
                "position_size": position_size,
                "stop_loss": stop_loss,
                "profit_target": profit_target,
                "hedge_enabled": hedge_enabled,
                "paper_trading": paper_trading,
                "created_at": datetime.now().isoformat()
            }
            
            save_strategy_config(config)
            st.success(f"✅ Greeks Strategy '{strategy_name}' configured successfully!")

def configure_custom_strategy():
    """Configure custom strategy"""
    st.subheader("🔧 Custom Strategy Configuration")
    
    st.info("Custom strategy configuration coming soon. This will allow you to define custom logic using a visual strategy builder.")
    
    # Placeholder for custom strategy builder
    st.write("**Features Coming Soon:**")
    st.write("- Visual strategy builder")
    st.write("- Custom indicator combinations")
    st.write("- Advanced condition logic")
    st.write("- Machine learning integration")

def show_backtest_results():
    """Show strategy backtest results"""
    st.subheader("📊 Strategy Backtest Results")
    
    # Backtest controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strategy_to_backtest = st.selectbox(
            "Select Strategy",
            get_available_strategies()
        )
    
    with col2:
        backtest_period = st.selectbox(
            "Backtest Period",
            ["1 Month", "3 Months", "6 Months", "1 Year"]
        )
    
    with col3:
        if st.button("🚀 Run Backtest", type="primary"):
            run_backtest(strategy_to_backtest, backtest_period)
    
    # Display backtest results if available
    if 'backtest_results' in st.session_state:
        results = st.session_state.backtest_results
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{results['total_return']:.1f}%")
        
        with col2:
            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{results['max_drawdown']:.1f}%")
        
        with col4:
            st.metric("Win Rate", f"{results['win_rate']:.1f}%")
        
        # Detailed results table
        st.subheader("📈 Performance Analysis")
        
        performance_df = pd.DataFrame(results['monthly_returns'])
        st.dataframe(performance_df, use_container_width=True)
        
        # Trade analysis
        st.subheader("🔍 Trade Analysis")
        
        trade_df = pd.DataFrame(results['trades'])
        st.dataframe(trade_df, use_container_width=True)

# Helper functions

def get_active_strategies_data() -> pd.DataFrame:
    """Get active strategies data"""
    # Sample data - in real implementation, fetch from database
    data = [
        {
            'Strategy': 'Breakout_NIFTY',
            'Status': 'Active',
            'Symbols': 'NIFTY CE/PE',
            'PnL': '₹2,450',
            'PnL_Change': '+₹340',
            'Trades': 12,
            'Win_Rate': '75%'
        },
        {
            'Strategy': 'OI_Analysis_BANKNIFTY',
            'Status': 'Active', 
            'Symbols': 'BANKNIFTY CE/PE',
            'PnL': '₹1,850',
            'PnL_Change': '+₹125',
            'Trades': 8,
            'Win_Rate': '62.5%'
        },
        {
            'Strategy': 'Greeks_Delta_Neutral',
            'Status': 'Paused',
            'Symbols': 'NIFTY CE/PE',
            'PnL': '-₹320',
            'PnL_Change': '-₹80',
            'Trades': 15,
            'Win_Rate': '46.7%'
        }
    ]
    
    return pd.DataFrame(data)

def toggle_strategy_status(strategy_name: str):
    """Toggle strategy active/paused status"""
    st.success(f"Strategy '{strategy_name}' status toggled!")
    st.rerun()

def save_strategy_config(config: dict):
    """Save strategy configuration"""
    # In real implementation, save to database
    if 'strategy_configs' not in st.session_state:
        st.session_state.strategy_configs = []
    
    st.session_state.strategy_configs.append(config)

def get_available_strategies() -> list:
    """Get list of available strategies for backtesting"""
    return [
        "Breakout_NIFTY",
        "OI_Analysis_BANKNIFTY", 
        "Greeks_Delta_Neutral",
        "Custom_Strategy_1"
    ]

def run_backtest(strategy: str, period: str):
    """Run backtest for selected strategy"""
    # Sample backtest results
    import random
    
    results = {
        'strategy': strategy,
        'period': period,
        'total_return': random.uniform(5, 25),
        'sharpe_ratio': random.uniform(0.8, 2.5),
        'max_drawdown': random.uniform(5, 15),
        'win_rate': random.uniform(55, 80),
        'monthly_returns': [
            {'Month': 'Jan 2024', 'Return': '5.2%', 'Trades': 25, 'Win Rate': '72%'},
            {'Month': 'Feb 2024', 'Return': '3.8%', 'Trades': 18, 'Win Rate': '67%'},
            {'Month': 'Mar 2024', 'Return': '-1.2%', 'Trades': 22, 'Win Rate': '45%'},
            {'Month': 'Apr 2024', 'Return': '7.1%', 'Trades': 28, 'Win Rate': '79%'}
        ],
        'trades': [
            {'Date': '2024-01-15', 'Symbol': 'NIFTY25JAN23C21500', 'Action': 'BUY', 'Price': 45.50, 'P&L': '+₹275'},
            {'Date': '2024-01-16', 'Symbol': 'NIFTY25JAN23C21500', 'Action': 'SELL', 'Price': 51.00, 'P&L': '+₹275'},
            {'Date': '2024-01-18', 'Symbol': 'NIFTY25JAN23P21400', 'Action': 'SELL', 'Price': 32.25, 'P&L': '+₹150'},
            {'Date': '2024-01-19', 'Symbol': 'NIFTY25JAN23P21400', 'Action': 'BUY', 'Price': 29.25, 'P&L': '+₹150'}
        ]
    }
    
    st.session_state.backtest_results = results
    st.success(f"✅ Backtest completed for {strategy} over {period}")
    st.rerun()
