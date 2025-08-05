import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from risk_management.risk_calculator import RiskCalculator
from risk_management.position_manager import PositionManager
from telegram_alerts import send_telegram_alert

st.set_page_config(page_title="Risk Monitor", layout="wide")

MODULE_NAME = "risk_monitor"

def show_risk_monitor():
    """Display comprehensive risk monitoring dashboard"""

    try:
        st.header("🛡️ Risk Management Monitor")

        # Risk overview metrics
        show_risk_overview()

        # Main risk monitoring tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Position Risk",
            "📊 Portfolio Risk",
            "⚠️ Risk Alerts",
            "🧪 Stress Testing"
        ])

        with tab1:
            show_position_risk()

        with tab2:
            show_portfolio_risk()

        with tab3:
            show_risk_alerts()

        with tab4:
            show_stress_testing()
    except Exception as exc:
        send_telegram_alert(f"{MODULE_NAME} error: {exc}")
        st.error("An error occurred while loading the Risk Management Monitor page.")

def show_risk_overview():
    """Display high-level risk metrics"""
    st.subheader("📈 Risk Overview")
    
    # Risk metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Portfolio VaR (95%)", "₹15,240", "-₹1,200")
    
    with col2:
        st.metric("Max Drawdown", "8.5%", "+1.2%")
    
    with col3:
        st.metric("Sharpe Ratio", "1.85", "+0.12")
    
    with col4:
        st.metric("Portfolio Beta", "1.23", "-0.05")
    
    with col5:
        st.metric("Risk Score", "Medium", "")
    
    # Risk status indicator
    risk_status = get_current_risk_status()
    
    if risk_status == "Low":
        st.success("🟢 **Risk Status: LOW** - Portfolio is within acceptable risk limits")
    elif risk_status == "Medium":
        st.warning("🟡 **Risk Status: MEDIUM** - Monitor closely for risk limit breaches")
    else:
        st.error("🔴 **Risk Status: HIGH** - Immediate attention required!")

def show_position_risk():
    """Display individual position risk analysis"""
    st.subheader("🎯 Position-Level Risk Analysis")
    
    # Position risk controls
    col1, col2 = st.columns([1, 3])
    
    with col1:
        sort_by = st.selectbox(
            "Sort Positions By",
            ["Risk Score", "P&L", "Position Size", "Time to Expiry"]
        )
        
        risk_filter = st.selectbox(
            "Risk Filter",
            ["All Positions", "High Risk", "Medium Risk", "Low Risk"]
        )
        
        show_greeks = st.checkbox("Show Greeks", value=True)
    
    with col2:
        # Position risk table
        positions_df = get_positions_risk_data()
        
        if not positions_df.empty:
            # Style the dataframe based on risk levels
            styled_df = style_position_risk_table(positions_df)
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Position risk alerts
            high_risk_positions = positions_df[positions_df['Risk_Score'] == 'High']
            if not high_risk_positions.empty:
                st.warning(f"⚠️ {len(high_risk_positions)} positions flagged as HIGH RISK")
                
                for _, pos in high_risk_positions.iterrows():
                    st.error(f"🚨 {pos['Symbol']}: {pos['Risk_Reason']}")
        else:
            st.info("No active positions to analyze")
    
    # Position Greeks analysis
    if show_greeks:
        show_position_greeks_analysis()

def show_portfolio_risk():
    """Display portfolio-level risk metrics"""
    st.subheader("📊 Portfolio Risk Analytics")
    
    # Portfolio risk metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Risk Metrics**")
        
        risk_metrics = get_portfolio_risk_metrics()
        
        metrics_df = pd.DataFrame([
            {"Metric": "Value at Risk (95%)", "Value": f"₹{risk_metrics['var_95']:,.0f}", "Limit": "₹50,000"},
            {"Metric": "Value at Risk (99%)", "Value": f"₹{risk_metrics['var_99']:,.0f}", "Limit": "₹75,000"},
            {"Metric": "Expected Shortfall", "Value": f"₹{risk_metrics['expected_shortfall']:,.0f}", "Limit": "₹60,000"},
            {"Metric": "Maximum Drawdown", "Value": f"{risk_metrics['max_drawdown']:.1f}%", "Limit": "15%"},
            {"Metric": "Portfolio Beta", "Value": f"{risk_metrics['beta']:.2f}", "Limit": "1.5"},
            {"Metric": "Correlation to Market", "Value": f"{risk_metrics['correlation']:.2f}", "Limit": "0.8"}
        ])
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**Risk Distribution**")
        
        # Risk contribution chart
        risk_contrib_data = get_risk_contribution_data()
        
        fig_risk = px.pie(
            risk_contrib_data, 
            values='Risk_Contribution', 
            names='Position',
            title="Risk Contribution by Position"
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Portfolio Greeks summary
    st.write("**Portfolio Greeks**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    portfolio_greeks = get_portfolio_greeks()
    
    with col1:
        delta_color = "normal" if abs(portfolio_greeks['delta']) < 50 else "inverse"
        st.metric("Total Delta", f"{portfolio_greeks['delta']:.1f}", delta_color=delta_color)
    
    with col2:
        st.metric("Total Gamma", f"{portfolio_greeks['gamma']:.3f}")
    
    with col3:
        st.metric("Total Theta", f"{portfolio_greeks['theta']:.0f}")
    
    with col4:
        st.metric("Total Vega", f"{portfolio_greeks['vega']:.1f}")
    
    with col5:
        st.metric("Total Rho", f"{portfolio_greeks['rho']:.2f}")
    
    # Historical risk chart
    st.write("**Historical Risk Trend**")
    
    historical_risk_chart = create_historical_risk_chart()
    st.plotly_chart(historical_risk_chart, use_container_width=True)

def show_risk_alerts():
    """Display risk alerts and warnings"""
    st.subheader("⚠️ Risk Alerts & Warnings")
    
    # Current alerts
    current_alerts = get_current_risk_alerts()
    
    if current_alerts:
        st.write("**🚨 Active Alerts**")
        
        for alert in current_alerts:
            if alert['severity'] == 'Critical':
                st.error(f"🔴 **{alert['type']}**: {alert['message']}")
            elif alert['severity'] == 'High':
                st.warning(f"🟡 **{alert['type']}**: {alert['message']}")
            else:
                st.info(f"🔵 **{alert['type']}**: {alert['message']}")
    else:
        st.success("✅ No active risk alerts")
    
    st.divider()
    
    # Risk limit monitoring
    st.write("**📊 Risk Limit Monitoring**")
    
    risk_limits_df = get_risk_limits_status()
    
    for _, limit in risk_limits_df.iterrows():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**{limit['Metric']}**")
        
        with col2:
            current_val = limit['Current']
            limit_val = limit['Limit']
            utilization = (current_val / limit_val) * 100 if limit_val > 0 else 0
            
            if utilization > 90:
                color = "red"
            elif utilization > 75:
                color = "orange"
            else:
                color = "green"
            
            st.metric("Current", f"{current_val:.1f}", f"{utilization:.1f}% of limit")
        
        with col3:
            st.write(f"Limit: {limit_val:.1f}")
            
            # Progress bar
            progress_val = min(utilization / 100, 1.0)
            st.progress(progress_val)
    
    st.divider()
    
    # Alert configuration
    st.write("**⚙️ Alert Configuration**")
    
    with st.expander("Configure Risk Alerts"):
        col1, col2 = st.columns(2)
        
        with col1:
            var_alert_threshold = st.slider("VaR Alert Threshold (₹)", 10000, 100000, 50000, 5000)
            drawdown_alert_threshold = st.slider("Drawdown Alert (%)", 5, 20, 10, 1)
            position_size_alert = st.slider("Position Size Alert (₹)", 50000, 500000, 200000, 25000)
        
        with col2:
            beta_alert_threshold = st.slider("Beta Alert Threshold", 0.5, 3.0, 1.5, 0.1)
            correlation_alert = st.slider("Correlation Alert", 0.5, 1.0, 0.8, 0.05)
            volatility_alert = st.slider("Volatility Alert (%)", 20, 80, 50, 5)
        
        if st.button("💾 Save Alert Settings"):
            save_alert_settings({
                'var_threshold': var_alert_threshold,
                'drawdown_threshold': drawdown_alert_threshold,
                'position_size_threshold': position_size_alert,
                'beta_threshold': beta_alert_threshold,
                'correlation_threshold': correlation_alert,
                'volatility_threshold': volatility_alert
            })
            st.success("Alert settings saved!")

def show_stress_testing():
    """Display stress testing scenarios and results"""
    st.subheader("🧪 Stress Testing & Scenario Analysis")
    
    # Stress test controls
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Scenario Selection**")
        
        scenario_type = st.selectbox(
            "Stress Test Scenario",
            ["Market Crash (-20%)", "Moderate Decline (-10%)", "Volatility Spike (+50%)", 
             "Interest Rate Shock (+2%)", "Custom Scenario"]
        )
        
        if scenario_type == "Custom Scenario":
            market_shock = st.slider("Market Movement (%)", -30, 30, 0, 1)
            volatility_shock = st.slider("Volatility Change (%)", -50, 100, 0, 5)
            interest_rate_shock = st.slider("Interest Rate Change (%)", -2, 5, 0, 0.25)
        else:
            market_shock, volatility_shock, interest_rate_shock = get_predefined_scenario(scenario_type)
        
        if st.button("🧪 Run Stress Test", type="primary"):
            run_stress_test(market_shock, volatility_shock, interest_rate_shock)
    
    with col2:
        st.write("**Stress Test Results**")
        
        if 'stress_test_results' in st.session_state:
            results = st.session_state.stress_test_results
            
            # Results summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Portfolio Impact", f"₹{results['total_impact']:,.0f}")
            
            with col2:
                impact_pct = (results['total_impact'] / results['current_value']) * 100
                st.metric("Impact %", f"{impact_pct:.1f}%")
            
            with col3:
                if results['total_impact'] < 0:
                    st.metric("Risk Status", "⚠️ Loss", f"₹{abs(results['total_impact']):,.0f}")
                else:
                    st.metric("Risk Status", "✅ Gain", f"₹{results['total_impact']:,.0f}")
            
            # Detailed results by position
            st.write("**Impact by Position**")
            
            impact_df = pd.DataFrame(results['position_impacts'])
            
            # Create impact chart
            fig_impact = px.bar(
                impact_df, 
                x='Symbol', 
                y='Impact',
                color='Impact',
                color_continuous_scale='RdYlGn',
                title="Stress Test Impact by Position"
            )
            
            st.plotly_chart(fig_impact, use_container_width=True)
            
            st.dataframe(impact_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run a stress test to see results")
    
    # Scenario analysis
    st.divider()
    
    st.write("**📊 Scenario Analysis History**")
    
    scenario_history = get_scenario_history()
    
    if scenario_history:
        history_df = pd.DataFrame(scenario_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No scenario analysis history available")

# Helper functions for risk monitoring

def get_current_risk_status() -> str:
    """Determine current overall risk status"""
    # Sample logic - in real implementation, calculate based on multiple factors
    return "Medium"

def get_positions_risk_data() -> pd.DataFrame:
    """Get position-level risk data"""
    # Sample data - in real implementation, fetch from position manager
    data = [
        {
            'Symbol': 'NIFTY25JAN23C21500',
            'Position_Size': '₹85,000',
            'Current_P&L': '+₹2,400',
            'P&L_%': '+2.82%',
            'Delta': '0.65',
            'Gamma': '0.023',
            'Theta': '-12.5',
            'Days_to_Expiry': 5,
            'Risk_Score': 'Medium',
            'Risk_Reason': 'High theta decay'
        },
        {
            'Symbol': 'BANKNIFTY25JAN22P45000',
            'Position_Size': '₹67,500',
            'Current_P&L': '-₹850',
            'P&L_%': '-1.26%',
            'Delta': '-0.42',
            'Gamma': '0.018',
            'Theta': '-8.3',
            'Days_to_Expiry': 2,
            'Risk_Score': 'High',
            'Risk_Reason': 'Expiring soon'
        },
        {
            'Symbol': 'NIFTY25FEB06C21600',
            'Position_Size': '₹125,000',
            'Current_P&L': '+₹1,750',
            'P&L_%': '+1.40%',
            'Delta': '0.55',
            'Gamma': '0.019',
            'Theta': '-6.8',
            'Days_to_Expiry': 12,
            'Risk_Score': 'Low',
            'Risk_Reason': 'Within limits'
        }
    ]
    
    return pd.DataFrame(data)

def style_position_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply styling to position risk table"""
    # In real Streamlit implementation, you would use styling functions
    return df

def show_position_greeks_analysis():
    """Show detailed Greeks analysis for positions"""
    st.write("**🏛️ Position Greeks Analysis**")
    
    greeks_data = get_position_greeks_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Delta exposure chart
        fig_delta = px.bar(
            greeks_data, 
            x='Symbol', 
            y='Delta',
            title="Delta Exposure by Position",
            color='Delta',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_delta, use_container_width=True)
    
    with col2:
        # Theta decay chart
        fig_theta = px.bar(
            greeks_data, 
            x='Symbol', 
            y='Theta',
            title="Theta Decay by Position",
            color='Theta',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_theta, use_container_width=True)

def get_portfolio_risk_metrics() -> dict:
    """Get portfolio-level risk metrics"""
    return {
        'var_95': 15240,
        'var_99': 22150,
        'expected_shortfall': 18500,
        'max_drawdown': 8.5,
        'beta': 1.23,
        'correlation': 0.68,
        'volatility': 24.5,
        'sharpe_ratio': 1.85
    }

def get_risk_contribution_data() -> pd.DataFrame:
    """Get risk contribution by position"""
    return pd.DataFrame([
        {'Position': 'NIFTY Options', 'Risk_Contribution': 45.2},
        {'Position': 'BANKNIFTY Options', 'Risk_Contribution': 38.7},
        {'Position': 'FINNIFTY Options', 'Risk_Contribution': 16.1}
    ])

def get_portfolio_greeks() -> dict:
    """Get portfolio-level Greeks"""
    return {
        'delta': 23.4,
        'gamma': 0.156,
        'theta': -145.7,
        'vega': 89.3,
        'rho': 12.8
    }

def create_historical_risk_chart() -> go.Figure:
    """Create historical risk trend chart"""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    var_values = np.random.normal(15000, 2000, 30)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=var_values,
        mode='lines+markers',
        name='VaR (95%)',
        line=dict(color='red', width=2)
    ))
    
    # Add limit line
    fig.add_hline(y=50000, line_dash="dash", line_color="orange", 
                  annotation_text="VaR Limit")
    
    fig.update_layout(
        title="Historical VaR Trend",
        xaxis_title="Date",
        yaxis_title="VaR (₹)",
        height=400
    )
    
    return fig

def get_current_risk_alerts() -> list:
    """Get current active risk alerts"""
    return [
        {
            'type': 'Position Concentration',
            'severity': 'High',
            'message': 'NIFTY options represent 65% of portfolio - consider diversification'
        },
        {
            'type': 'Expiry Risk',
            'severity': 'Critical',
            'message': 'BANKNIFTY25JAN22P45000 expires in 2 days with negative P&L'
        }
    ]

def get_risk_limits_status() -> pd.DataFrame:
    """Get current status vs risk limits"""
    return pd.DataFrame([
        {'Metric': 'Portfolio VaR', 'Current': 15240, 'Limit': 50000, 'Status': 'OK'},
        {'Metric': 'Max Drawdown', 'Current': 8.5, 'Limit': 15.0, 'Status': 'OK'},
        {'Metric': 'Position Concentration', 'Current': 65.0, 'Limit': 70.0, 'Status': 'Warning'},
        {'Metric': 'Portfolio Beta', 'Current': 1.23, 'Limit': 1.50, 'Status': 'OK'}
    ])

def save_alert_settings(settings: dict):
    """Save alert configuration settings"""
    # In real implementation, save to database
    st.session_state.alert_settings = settings

def get_predefined_scenario(scenario_name: str) -> tuple:
    """Get parameters for predefined scenarios"""
    scenarios = {
        "Market Crash (-20%)": (-20, 50, 0),
        "Moderate Decline (-10%)": (-10, 25, 0),
        "Volatility Spike (+50%)": (0, 50, 0),
        "Interest Rate Shock (+2%)": (0, 10, 2)
    }
    
    return scenarios.get(scenario_name, (0, 0, 0))

def run_stress_test(market_shock: float, volatility_shock: float, interest_rate_shock: float):
    """Run stress test with given parameters"""
    # Sample stress test results
    results = {
        'scenario': f"Market: {market_shock}%, Vol: {volatility_shock}%, IR: {interest_rate_shock}%",
        'current_value': 275000,
        'total_impact': market_shock * 2500 + volatility_shock * 150,  # Simplified calculation
        'position_impacts': [
            {'Symbol': 'NIFTY25JAN23C21500', 'Current_Value': 85000, 'Impact': market_shock * 850},
            {'Symbol': 'BANKNIFTY25JAN22P45000', 'Current_Value': 67500, 'Impact': market_shock * 675},
            {'Symbol': 'NIFTY25FEB06C21600', 'Current_Value': 125000, 'Impact': market_shock * 1250}
        ]
    }
    
    st.session_state.stress_test_results = results
    st.rerun()

def get_scenario_history() -> list:
    """Get historical scenario analysis results"""
    return [
        {
            'Date': '2024-01-15',
            'Scenario': 'Market Crash (-20%)',
            'Impact': '-₹55,000',
            'Impact_%': '-20.0%'
        },
        {
            'Date': '2024-01-10',
            'Scenario': 'Volatility Spike (+50%)',
            'Impact': '+₹7,500',
            'Impact_%': '+2.7%'
        }
    ]

def get_position_greeks_data() -> pd.DataFrame:
    """Get Greeks data for chart visualization"""
    return pd.DataFrame([
        {'Symbol': 'NIFTY_C21500', 'Delta': 0.65, 'Gamma': 0.023, 'Theta': -12.5, 'Vega': 18.2},
        {'Symbol': 'BANKNIFTY_P45000', 'Delta': -0.42, 'Gamma': 0.018, 'Theta': -8.3, 'Vega': 15.6},
        {'Symbol': 'NIFTY_C21600', 'Delta': 0.55, 'Gamma': 0.019, 'Theta': -6.8, 'Vega': 12.4}
    ])
