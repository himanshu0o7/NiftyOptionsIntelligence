import sqlite3
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from core.database import Database
from telegram_alerts import send_telegram_alert
from utils.helpers import helper
from utils.logger import Logger
from telegram_alerts import send_telegram_alert

st.set_page_config(page_title="P&L Analysis", layout="wide")

MODULE_NAME = "pnl_analysis"

MODULE_NAME = "pnl_analysis"

def show_pnl_analysis():
    """Display comprehensive P&L analysis dashboard"""

    try:
        st.header("📈 P&L Analysis & Performance")

        # P&L overview metrics
        show_pnl_overview()

        # Main P&L analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Daily P&L",
            "📈 Performance Trends",
            "🎯 Strategy Performance",
            "📋 Trade Analysis"
        ])

        with tab1:
            show_daily_pnl()

        with tab2:
            show_performance_trends()

        with tab3:
            show_strategy_performance()

        with tab4:
            show_trade_analysis()
    except Exception as exc:
        tb = traceback.format_exc()
        send_telegram_alert(f"{MODULE_NAME} error: {exc}\nTraceback:\n{tb}")
        st.error("An error occurred while loading the P&L Analysis page.")

def show_pnl_overview():
    """Display high-level P&L metrics"""
    st.subheader("💰 P&L Overview")
    
    # Time period selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        period = st.selectbox(
            "Analysis Period",
            ["Today", "This Week", "This Month", "Last 3 Months", "YTD", "All Time"]
        )
    
    with col2:
        currency = st.selectbox("Currency", ["INR (₹)", "USD ($)"], index=0)
    
    with col3:
        if st.button("🔄 Refresh Data"):
            refresh_pnl_data()
    
    # Key P&L metrics
    pnl_metrics = get_pnl_metrics(period)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_pnl = pnl_metrics['total_pnl']
        pnl_change = pnl_metrics['pnl_change']
        delta_color = "normal" if total_pnl >= 0 else "inverse"
        st.metric(
            "Total P&L", 
            helper.format_currency(total_pnl), 
            helper.format_currency(pnl_change),
            delta_color=delta_color
        )
    
    with col2:
        realized_pnl = pnl_metrics['realized_pnl']
        st.metric("Realized P&L", helper.format_currency(realized_pnl))
    
    with col3:
        unrealized_pnl = pnl_metrics['unrealized_pnl']
        st.metric("Unrealized P&L", helper.format_currency(unrealized_pnl))
    
    with col4:
        win_rate = pnl_metrics['win_rate']
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col5:
        avg_trade = pnl_metrics['avg_trade_pnl']
        st.metric("Avg Trade P&L", helper.format_currency(avg_trade))
    
    # P&L status indicator
    if total_pnl > 0:
        st.success(f"🟢 **Profitable Period** - Total gain of {helper.format_currency(total_pnl)}")
    elif total_pnl == 0:
        st.info("⚪ **Breakeven** - No net profit or loss")
    else:
        st.error(f"🔴 **Loss Period** - Total loss of {helper.format_currency(abs(total_pnl))}")

def show_daily_pnl():
    """Display daily P&L breakdown and analysis"""
    st.subheader("📊 Daily P&L Analysis")
    
    # Date range selector
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=30)
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date()
        )
    
    # Daily P&L chart
    daily_pnl_data = get_daily_pnl_data(start_date, end_date)
    
    if not daily_pnl_data.empty:
        # Create daily P&L chart
        fig_daily = create_daily_pnl_chart(daily_pnl_data)
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Daily statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            profitable_days = len(daily_pnl_data[daily_pnl_data['daily_pnl'] > 0])
            total_days = len(daily_pnl_data)
            profitable_pct = (profitable_days / total_days * 100) if total_days > 0 else 0
            st.metric("Profitable Days", f"{profitable_days}/{total_days}", f"{profitable_pct:.1f}%")
        
        with col2:
            best_day = daily_pnl_data['daily_pnl'].max()
            st.metric("Best Day", helper.format_currency(best_day))
        
        with col3:
            worst_day = daily_pnl_data['daily_pnl'].min()
            st.metric("Worst Day", helper.format_currency(worst_day))
        
        with col4:
            avg_daily = daily_pnl_data['daily_pnl'].mean()
            st.metric("Avg Daily P&L", helper.format_currency(avg_daily))
        
        # Daily P&L table
        st.subheader("📋 Daily P&L Details")
        
        # Add additional columns for analysis
        daily_pnl_data['cumulative_pnl'] = daily_pnl_data['daily_pnl'].cumsum()
        daily_pnl_data['trades_count'] = daily_pnl_data.get('trades_count', 0)
        daily_pnl_data['win_rate_daily'] = daily_pnl_data.get('win_rate', 0)
        
        # Format for display
        display_data = daily_pnl_data.copy()
        display_data['Date'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
        display_data['Daily P&L'] = display_data['daily_pnl'].apply(helper.format_currency)
        display_data['Cumulative P&L'] = display_data['cumulative_pnl'].apply(helper.format_currency)
        display_data['Trades'] = display_data['trades_count']
        display_data['Win Rate'] = display_data['win_rate_daily'].apply(lambda x: f"{x:.1f}%" if x > 0 else "-")
        
        st.dataframe(
            display_data[['Date', 'Daily P&L', 'Cumulative P&L', 'Trades', 'Win Rate']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No P&L data available for the selected date range")

def show_performance_trends():
    """Display performance trends and metrics over time"""
    st.subheader("📈 Performance Trends")
    
    # Performance period selector
    trend_period = st.selectbox(
        "Trend Analysis Period",
        ["Last 30 Days", "Last 3 Months", "Last 6 Months", "Last Year"],
        index=1
    )
    
    # Get performance data
    performance_data = get_performance_trends_data(trend_period)
    
    if not performance_data.empty:
        # Performance metrics over time
        col1, col2 = st.columns(2)
        
        with col1:
            # Cumulative P&L chart
            fig_cumulative = create_cumulative_pnl_chart(performance_data)
            st.plotly_chart(fig_cumulative, use_container_width=True)
        
        with col2:
            # Rolling Sharpe ratio chart
            fig_sharpe = create_rolling_sharpe_chart(performance_data)
            st.plotly_chart(fig_sharpe, use_container_width=True)
        
        # Monthly performance breakdown
        st.subheader("📅 Monthly Performance Breakdown")
        
        monthly_data = get_monthly_performance_data(trend_period)
        
        if not monthly_data.empty:
            # Monthly P&L heatmap
            fig_heatmap = create_monthly_heatmap(monthly_data)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Monthly statistics table
            monthly_stats = calculate_monthly_stats(monthly_data)
            st.dataframe(monthly_stats, use_container_width=True, hide_index=True)
        
        # Risk-Return Analysis
        st.subheader("⚖️ Risk-Return Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        risk_metrics = calculate_risk_return_metrics(performance_data)
        
        with col1:
            st.metric("Sharpe Ratio", f"{risk_metrics['sharpe_ratio']:.2f}")
            st.metric("Sortino Ratio", f"{risk_metrics['sortino_ratio']:.2f}")
        
        with col2:
            st.metric("Max Drawdown", f"{risk_metrics['max_drawdown']:.1f}%")
            st.metric("Volatility", f"{risk_metrics['volatility']:.1f}%")
        
        with col3:
            st.metric("Calmar Ratio", f"{risk_metrics['calmar_ratio']:.2f}")
            st.metric("Profit Factor", f"{risk_metrics['profit_factor']:.2f}")
    else:
        st.info("No performance data available for the selected period")

def show_strategy_performance():
    """Display performance breakdown by strategy"""
    st.subheader("🎯 Strategy Performance Analysis")
    
    # Strategy performance overview
    strategy_data = get_strategy_performance_data()
    
    if not strategy_data.empty:
        # Strategy comparison chart
        fig_strategy = create_strategy_comparison_chart(strategy_data)
        st.plotly_chart(fig_strategy, use_container_width=True)
        
        # Strategy performance table
        st.subheader("📊 Strategy Performance Summary")
        
        # Format strategy data for display
        display_strategy_data = format_strategy_performance_data(strategy_data)
        st.dataframe(display_strategy_data, use_container_width=True, hide_index=True)
        
        # Individual strategy analysis
        st.subheader("🔍 Individual Strategy Analysis")
        
        selected_strategy = st.selectbox(
            "Select Strategy for Detailed Analysis",
            strategy_data['strategy_name'].unique().tolist()
        )
        
        if selected_strategy:
            show_individual_strategy_analysis(selected_strategy)
    else:
        st.info("No strategy performance data available")

def show_trade_analysis():
    """Display detailed trade analysis"""
    st.subheader("📋 Trade Analysis")
    
    # Trade filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trade_period = st.selectbox(
            "Period",
            ["Today", "Last 7 Days", "Last 30 Days", "All Trades"]
        )
    
    with col2:
        trade_status = st.selectbox(
            "Trade Status",
            ["All", "Profitable", "Loss-making", "Breakeven"]
        )
    
    with col3:
        symbol_filter = st.selectbox(
            "Symbol",
            ["All Symbols", "NIFTY", "BANKNIFTY", "Others"]
        )
    
    # Get trade data
    trades_data = get_trades_data(trade_period, trade_status, symbol_filter)
    
    if not trades_data.empty:
        # Trade statistics
        col1, col2, col3, col4 = st.columns(4)
        
        trade_stats = calculate_trade_statistics(trades_data)
        
        with col1:
            st.metric("Total Trades", trade_stats['total_trades'])
        
        with col2:
            st.metric("Winning Trades", f"{trade_stats['winning_trades']} ({trade_stats['win_rate']:.1f}%)")
        
        with col3:
            st.metric("Avg Win", helper.format_currency(trade_stats['avg_win']))
        
        with col4:
            st.metric("Avg Loss", helper.format_currency(trade_stats['avg_loss']))
        
        # Trade distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pnl_dist = create_trade_pnl_distribution(trades_data)
            st.plotly_chart(fig_pnl_dist, use_container_width=True)
        
        with col2:
            fig_duration_analysis = create_trade_duration_analysis(trades_data)
            st.plotly_chart(fig_duration_analysis, use_container_width=True)
        
        # Detailed trades table
        st.subheader("📄 Trade Details")
        
        # Format trades data for display
        display_trades_data = format_trades_data(trades_data)
        
        # Add pagination for large datasets
        page_size = 50
        total_trades = len(display_trades_data)
        total_pages = (total_trades - 1) // page_size + 1
        
        if total_pages > 1:
            page = st.selectbox(f"Page (Total: {total_pages})", range(1, total_pages + 1))
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_trades)
            display_trades_data = display_trades_data.iloc[start_idx:end_idx]
        
        st.dataframe(display_trades_data, use_container_width=True, hide_index=True)
        
        # Trade export option
        if st.button("📥 Export Trade Data"):
            export_trade_data(trades_data)
    else:
        st.info("No trades found for the selected criteria")

# Helper functions for P&L analysis

def refresh_pnl_data():
    """Refresh P&L data from database"""
    st.success("P&L data refreshed successfully!")
    st.rerun()

def get_pnl_metrics(period: str) -> Dict:
    """Get P&L metrics for specified period"""
    db = Database()
    logger = Logger(MODULE_NAME)

    try:
        end_date = datetime.now().date()
        period = period.lower()
        if period == "today":
            start_date = end_date
        elif period == "this week":
            start_date = end_date - timedelta(days=end_date.weekday())
        elif period == "this month":
            start_date = end_date.replace(day=1)
        elif period == "last 3 months":
            start_date = end_date - timedelta(days=90)
        elif period == "ytd":
            start_date = end_date.replace(month=1, day=1)
        else:
            start_date = datetime(1970, 1, 1).date()

        conn = sqlite3.connect(db.db_path)

        summary_query = (
            "SELECT COALESCE(SUM(total_pnl),0) AS total_pnl, "
            "COALESCE(SUM(total_trades),0) AS total_trades, "
            "COALESCE(SUM(winning_trades),0) AS winning_trades "
            "FROM daily_summary WHERE date BETWEEN ? AND ?"
        )
        df = pd.read_sql_query(summary_query, conn, params=(str(start_date), str(end_date)))

        realized_pnl = float(df["total_pnl"].iloc[0]) if not df.empty else 0.0
        total_trades = int(df["total_trades"].iloc[0]) if not df.empty else 0
        winning_trades = int(df["winning_trades"].iloc[0]) if not df.empty else 0

        unrealized_query = (
            "SELECT COALESCE(SUM(unrealized_pnl),0) AS unrealized FROM positions "
            "WHERE status = 'OPEN'"
        )
        unrealized_df = pd.read_sql_query(unrealized_query, conn)
        unrealized_pnl = float(unrealized_df["unrealized"].iloc[0]) if not unrealized_df.empty else 0.0

        change_query = (
            "SELECT total_pnl FROM daily_summary ORDER BY date DESC LIMIT 2"
        )
        change_df = pd.read_sql_query(change_query, conn)
        pnl_change = 0.0
        if len(change_df) == 2:
            pnl_change = float(change_df.iloc[0]["total_pnl"] - change_df.iloc[1]["total_pnl"])

        conn.close()

        win_rate = (winning_trades / total_trades * 100) if total_trades else 0
        avg_trade_pnl = (realized_pnl / total_trades) if total_trades else 0
        total_pnl = realized_pnl + unrealized_pnl

        return {
            "total_pnl": round(total_pnl, 2),
            "pnl_change": round(pnl_change, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "win_rate": round(win_rate, 2),
            "avg_trade_pnl": round(avg_trade_pnl, 2),
        }
    except Exception as e:
        error_msg = f"{MODULE_NAME} get_pnl_metrics error: {e}"
        logger.error(error_msg)
        send_telegram_alert(error_msg)
        return {
            "total_pnl": 0,
            "pnl_change": 0,
            "realized_pnl": 0,
            "unrealized_pnl": 0,
            "win_rate": 0,
            "avg_trade_pnl": 0,
        }

def get_daily_pnl_data(start_date, end_date) -> pd.DataFrame:
    """Get daily P&L data for date range"""
    db = Database()
    logger = Logger(MODULE_NAME)

    try:
        conn = sqlite3.connect(db.db_path)
        query = (
            "SELECT date, total_pnl, total_trades, win_rate "
            "FROM daily_summary WHERE date BETWEEN ? AND ? ORDER BY date"
        )
        df = pd.read_sql_query(query, conn, params=(str(start_date), str(end_date)))
        conn.close()

        if df.empty:
            return pd.DataFrame()

        df.rename(
            columns={
                "total_pnl": "daily_pnl",
                "total_trades": "trades_count",
                "win_rate": "win_rate",
            },
            inplace=True,
        )
        df["date"] = pd.to_datetime(df["date"])
        return df

    except Exception as e:
        error_msg = f"{MODULE_NAME} get_daily_pnl_data error: {e}"
        logger.error(error_msg)
        send_telegram_alert(error_msg)
        return pd.DataFrame()

def create_daily_pnl_chart(data: pd.DataFrame) -> go.Figure:
    """Create daily P&L chart"""
    fig = go.Figure()
    
    # Daily P&L bars
    colors = ['green' if pnl >= 0 else 'red' for pnl in data['daily_pnl']]
    
    fig.add_trace(go.Bar(
        x=data['date'],
        y=data['daily_pnl'],
        name='Daily P&L',
        marker_color=colors,
        text=[helper.format_currency(pnl) for pnl in data['daily_pnl']],
        textposition='outside'
    ))
    
    # Add cumulative P&L line
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['cumulative_pnl'],
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='blue', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Daily P&L Analysis",
        xaxis_title="Date",
        yaxis_title="Daily P&L (₹)",
        yaxis2=dict(
            title="Cumulative P&L (₹)",
            overlaying='y',
            side='right'
        ),
        height=500,
        hovermode='x unified'
    )
    
    return fig

def get_performance_trends_data(period: str) -> pd.DataFrame:
    """Get performance trends data"""
    db = Database()
    logger = Logger(MODULE_NAME)

    period_map = {
        "last 30 days": 30,
        "last 3 months": 90,
        "last 6 months": 180,
        "last year": 365,
    }

    try:
        days = period_map.get(period.lower(), 90)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days - 1)

        query = (
            "SELECT date, total_pnl FROM daily_summary "
            "WHERE date BETWEEN ? AND ? ORDER BY date"
        )
        with sqlite3.connect(db.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(str(start_date), str(end_date)))
        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        df.rename(columns={"total_pnl": "daily_pnl"}, inplace=True)
        df["cumulative_pnl"] = df["daily_pnl"].cumsum()
        max_abs = max(abs(df["daily_pnl"]).max(), 1)
        df["daily_returns"] = df["daily_pnl"] / max_abs
        return df

    except Exception as e:
        error_msg = f"{MODULE_NAME} get_performance_trends_data error: {e}"
        logger.error(error_msg)
        send_telegram_alert(error_msg)
        return pd.DataFrame()

def create_cumulative_pnl_chart(data: pd.DataFrame) -> go.Figure:
    """Create cumulative P&L chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['cumulative_pnl'],
        mode='lines',
        name='Cumulative P&L',
        line=dict(color='blue', width=3),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title="Cumulative P&L Trend",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L (₹)",
        height=400
    )
    
    return fig

def create_rolling_sharpe_chart(data: pd.DataFrame) -> go.Figure:
    """Create rolling Sharpe ratio chart"""
    fig = go.Figure()
    
    # Calculate rolling Sharpe ratio (simplified)
    if 'daily_returns' in data.columns:
        rolling_sharpe = data['daily_returns'].rolling(window=30).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=rolling_sharpe,
            mode='lines',
            name='30-Day Rolling Sharpe',
            line=dict(color='orange', width=2)
        ))
    
    fig.update_layout(
        title="Rolling Sharpe Ratio (30-Day)",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        height=400
    )
    
    return fig

def get_monthly_performance_data(period: str) -> pd.DataFrame:
    """Get monthly performance data"""
    # In real implementation, fetch from database
    return pd.DataFrame()

def create_monthly_heatmap(data: pd.DataFrame) -> go.Figure:
    """Create monthly performance heatmap"""
    fig = go.Figure()
    
    # Sample heatmap data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    years = ['2023', '2024']
    
    # Sample data matrix
    z = [[5.2, 3.8, -1.2, 7.1, 2.4, 4.6, 3.2, 1.8, 6.3, 2.1, 4.5, 3.7],
         [2.8, 4.1, 5.5, 3.2, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    fig.add_trace(go.Heatmap(
        z=z,
        x=months,
        y=years,
        colorscale='RdYlGn',
        text=[[f"{val}%" if val != 0 else "" for val in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Monthly Returns Heatmap",
        height=300
    )
    
    return fig

def calculate_monthly_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly statistics"""
    # Sample monthly stats
    return pd.DataFrame([
        {'Month': 'Jan 2024', 'P&L': '₹2,850', 'Trades': 45, 'Win Rate': '67%', 'Best Day': '₹485', 'Worst Day': '-₹320'},
        {'Month': 'Dec 2023', 'P&L': '₹4,120', 'Trades': 52, 'Win Rate': '73%', 'Best Day': '₹650', 'Worst Day': '-₹280'},
        {'Month': 'Nov 2023', 'P&L': '₹1,950', 'Trades': 38, 'Win Rate': '61%', 'Best Day': '₹420', 'Worst Day': '-₹380'}
    ])

def calculate_risk_return_metrics(data: pd.DataFrame) -> Dict:
    """Calculate risk-return metrics"""
    # Sample metrics - in real implementation, calculate from actual data
    return {
        'sharpe_ratio': 1.85,
        'sortino_ratio': 2.34,
        'max_drawdown': 8.5,
        'volatility': 24.3,
        'calmar_ratio': 1.42,
        'profit_factor': 1.68
    }

def get_strategy_performance_data() -> pd.DataFrame:
    """Get strategy performance data"""
    # In real implementation, fetch from database
    return pd.DataFrame()

def create_strategy_comparison_chart(data: pd.DataFrame) -> go.Figure:
    """Create strategy comparison chart"""
    # Sample strategy data
    strategies = ['Breakout Strategy', 'OI Analysis', 'Greeks Based']
    pnl_values = [8450, 6780, -1230]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=strategies,
        y=pnl_values,
        marker_color=['green' if p >= 0 else 'red' for p in pnl_values],
        text=[helper.format_currency(p) for p in pnl_values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Strategy Performance Comparison",
        xaxis_title="Strategy",
        yaxis_title="P&L (₹)",
        height=400
    )
    
    return fig

def format_strategy_performance_data(data: pd.DataFrame) -> pd.DataFrame:
    """Format strategy performance data for display"""
    # Sample formatted data
    return pd.DataFrame([
        {
            'Strategy': 'Breakout Strategy',
            'Total P&L': '₹8,450',
            'Win Rate': '72%',
            'Trades': 124,
            'Avg Trade': '₹68',
            'Max DD': '6.2%',
            'Sharpe': '1.92'
        },
        {
            'Strategy': 'OI Analysis',
            'Total P&L': '₹6,780',
            'Win Rate': '68%',
            'Trades': 89,
            'Avg Trade': '₹76',
            'Max DD': '8.1%',
            'Sharpe': '1.54'
        },
        {
            'Strategy': 'Greeks Based',
            'Total P&L': '-₹1,230',
            'Win Rate': '45%',
            'Trades': 67,
            'Avg Trade': '-₹18',
            'Max DD': '12.3%',
            'Sharpe': '0.23'
        }
    ])

def show_individual_strategy_analysis(strategy_name: str):
    """Show detailed analysis for individual strategy"""
    st.write(f"**Detailed Analysis: {strategy_name}**")
    
    # Strategy-specific metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Strategy P&L", "₹8,450", "+₹1,230")
        st.metric("Active Positions", "5", "-1")
    
    with col2:
        st.metric("Win Rate", "72%", "+3%")
        st.metric("Avg Trade Duration", "2.3 hrs", "-0.5 hrs")
    
    with col3:
        st.metric("Risk Score", "Medium", "")
        st.metric("Next Signal", "2 mins", "")
    
    # Strategy performance chart
    strategy_chart = create_individual_strategy_chart(strategy_name)
    st.plotly_chart(strategy_chart, use_container_width=True)

def create_individual_strategy_chart(strategy_name: str) -> go.Figure:
    """Create individual strategy performance chart"""
    # Sample data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    cumulative_pnl = np.cumsum(np.random.normal(50, 200, 30))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative_pnl,
        mode='lines+markers',
        name=f'{strategy_name} P&L',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=f"{strategy_name} - Performance Trend",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L (₹)",
        height=400
    )
    
    return fig

def get_trades_data(period: str, status: str, symbol: str) -> pd.DataFrame:
    """Get trades data based on filters"""
    # In real implementation, fetch from database with filters
    db = Database()
    logger = Logger()
    
    try:
        # Return empty DataFrame if no data
        # In real implementation: return db.get_trades(period, status, symbol)
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error fetching trades data: {str(e)}")
        return pd.DataFrame()

def calculate_trade_statistics(trades_data: pd.DataFrame) -> Dict:
    """Calculate trade statistics"""
    if trades_data.empty:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0
        }
    
    total_trades = len(trades_data)
    winning_trades = len(trades_data[trades_data['pnl'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    wins = trades_data[trades_data['pnl'] > 0]['pnl']
    losses = trades_data[trades_data['pnl'] < 0]['pnl']
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }

def create_trade_pnl_distribution(trades_data: pd.DataFrame) -> go.Figure:
    """Create trade P&L distribution chart"""
    fig = go.Figure()
    
    if not trades_data.empty and 'pnl' in trades_data.columns:
        fig.add_trace(go.Histogram(
            x=trades_data['pnl'],
            nbinsx=20,
            name='Trade P&L Distribution',
            marker_color='blue',
            opacity=0.7
        ))
    
    fig.update_layout(
        title="Trade P&L Distribution",
        xaxis_title="P&L (₹)",
        yaxis_title="Number of Trades",
        height=400
    )
    
    return fig

def create_trade_duration_analysis(trades_data: pd.DataFrame) -> go.Figure:
    """Create trade duration analysis chart"""
    fig = go.Figure()
    
    # Sample duration data
    durations = ['< 1 hr', '1-2 hrs', '2-4 hrs', '4-8 hrs', '> 8 hrs']
    counts = [15, 25, 18, 12, 8]
    
    fig.add_trace(go.Bar(
        x=durations,
        y=counts,
        name='Trade Duration',
        marker_color='orange'
    ))
    
    fig.update_layout(
        title="Trade Duration Analysis",
        xaxis_title="Duration",
        yaxis_title="Number of Trades",
        height=400
    )
    
    return fig

def format_trades_data(trades_data: pd.DataFrame) -> pd.DataFrame:
    """Format trades data for display"""
    if trades_data.empty:
        return pd.DataFrame()
    
    # Format the data for better display
    display_data = trades_data.copy()
    
    # Add formatted columns if they exist
    if 'pnl' in display_data.columns:
        display_data['P&L'] = display_data['pnl'].apply(helper.format_currency)
    
    if 'entry_time' in display_data.columns:
        display_data['Entry Time'] = pd.to_datetime(display_data['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
    
    if 'exit_time' in display_data.columns:
        display_data['Exit Time'] = pd.to_datetime(display_data['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
    
    return display_data

def export_trade_data(trades_data: pd.DataFrame):
    """Export trade data to CSV"""
    if not trades_data.empty:
        csv_data = trades_data.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        st.success("Trade data prepared for download!")
    else:
        st.warning("No trade data to export")
