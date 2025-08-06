"""
Popup Manager for Enhanced Dashboard Experience
Handles detailed information popups and windows
"""
import streamlit as st
from typing import Dict, Any, List
from datetime import datetime

class PopupManager:
    """Manage popup windows and detailed information displays"""

    @staticmethod
    def show_signal_details(signal: Dict, title: str = "Signal Details"):
        """Show detailed signal information in popup"""
        with st.expander(f"🔍 {title}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Signal Information**")
                st.write(f"**Type:** {signal.get('signal_type', 'N/A')}")
                st.write(f"**Action:** {signal.get('action', 'N/A')}")
                st.write(f"**Confidence:** {signal.get('confidence', 0):.2%}")
                st.write(f"**Symbol:** {signal.get('symbol', 'N/A')}")

            with col2:
                st.markdown("**Timing & Method**")
                st.write(f"**Timestamp:** {signal.get('timestamp', 'N/A')}")
                st.write(f"**Method:** {signal.get('method', 'N/A')}")
                st.write(f"**Strike:** {signal.get('strike', 'N/A')}")
                st.write(f"**Expiry:** {signal.get('expiry', 'N/A')}")

            if 'features_used' in signal:
                st.markdown("**Features Used:**")
                st.write(", ".join(signal['features_used']))

            if 'reasoning' in signal:
                st.markdown("**AI Reasoning:**")
                st.info(signal['reasoning'])

    @staticmethod
    def show_position_details(position: Dict, title: str = "Position Details"):
        """Show detailed position information in popup"""
        with st.expander(f"📈 {title}", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Position Info**")
                st.write(f"**Symbol:** {position.get('symbol', 'N/A')}")
                st.write(f"**Quantity:** {position.get('quantity', 0)}")
                st.write(f"**Side:** {position.get('side', 'N/A')}")
                st.write(f"**Status:** {position.get('status', 'N/A')}")

            with col2:
                st.markdown("**Price & P&L**")
                st.write(f"**Entry Price:** ₹{position.get('entry_price', 0):.2f}")
                st.write(f"**Current Price:** ₹{position.get('current_price', 0):.2f}")
                st.write(f"**P&L:** ₹{position.get('pnl', 0):.2f}")
                st.write(f"**P&L %:** {position.get('pnl_percent', 0):.2f}%")

            with col3:
                st.markdown("**Risk Management**")
                st.write(f"**Stop Loss:** ₹{position.get('stop_loss', 0):.2f}")
                st.write(f"**Target:** ₹{position.get('target', 0):.2f}")
                st.write(f"**Greeks Delta:** {position.get('delta', 0):.4f}")
                st.write(f"**Time to Expiry:** {position.get('tte', 'N/A')}")

    @staticmethod
    def show_strategy_details(strategy: Dict, title: str = "Strategy Details"):
        """Show detailed strategy information in popup"""
        with st.expander(f"🎯 {title}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Strategy Configuration**")
                st.write(f"**Name:** {strategy.get('name', 'N/A')}")
                st.write(f"**Market Mode:** {strategy.get('market_mode', 'N/A')}")
                st.write(f"**Index:** {strategy.get('index', 'N/A')}")
                st.write(f"**Entry Type:** {strategy.get('entry_type', 'N/A')}")

            with col2:
                st.markdown("**Performance Metrics**")
                st.write(f"**Success Rate:** {strategy.get('success_rate', 0):.1f}%")
                st.write(f"**Total Trades:** {strategy.get('total_trades', 0)}")
                st.write(f"**Avg P&L:** ₹{strategy.get('avg_pnl', 0):.2f}")
                st.write(f"**Max Drawdown:** {strategy.get('max_drawdown', 0):.2f}%")

            st.markdown("**Strategy Description**")
            st.info(strategy.get('description', 'No description available'))

            if 'parameters' in strategy:
                st.markdown("**Parameters**")
                for key, value in strategy['parameters'].items():
                    st.write(f"**{key}:** {value}")

    @staticmethod
    def show_ml_model_details(model: Dict, title: str = "ML Model Details"):
        """Show detailed ML model information in popup"""
        with st.expander(f"🧠 {title}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Model Information**")
                st.write(f"**Model Type:** {model.get('type', 'N/A')}")
                st.write(f"**Accuracy:** {model.get('accuracy', 0):.3f}")
                st.write(f"**Precision:** {model.get('precision', 0):.3f}")
                st.write(f"**Recall:** {model.get('recall', 0):.3f}")
                st.write(f"**F1 Score:** {model.get('f1_score', 0):.3f}")

            with col2:
                st.markdown("**Training Information**")
                st.write(f"**Training Samples:** {model.get('training_samples', 0)}")
                st.write(f"**Features Used:** {model.get('features_count', 0)}")
                st.write(f"**Last Trained:** {model.get('last_trained', 'N/A')}")
                st.write(f"**Training Time:** {model.get('training_time', 'N/A')}")

            if 'feature_importance' in model:
                st.markdown("**Feature Importance**")
                features = model['feature_importance']
                for feature, importance in features.items():
                    st.write(f"**{feature}:** {importance:.3f}")

    @staticmethod
    def show_live_data_popup(data: Dict, title: str = "Live Data"):
        """Show live data in detailed popup"""
        with st.expander(f"📡 {title}", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Market Data**")
                st.write(f"**Symbol:** {data.get('symbol', 'N/A')}")
                st.write(f"**LTP:** ₹{data.get('ltp', 0):.2f}")
                st.write(f"**Change:** {data.get('change', 0):+.2f}")
                st.write(f"**Volume:** {data.get('volume', 0):,}")

            with col2:
                st.markdown("**Options Greeks**")
                st.write(f"**Delta:** {data.get('delta', 0):.4f}")
                st.write(f"**Gamma:** {data.get('gamma', 0):.4f}")
                st.write(f"**Theta:** {data.get('theta', 0):.4f}")
                st.write(f"**Vega:** {data.get('vega', 0):.4f}")

            with col3:
                st.markdown("**Market Sentiment**")
                st.write(f"**IV:** {data.get('iv', 0):.2f}%")
                st.write(f"**OI:** {data.get('oi', 0):,}")
                st.write(f"**OI Change:** {data.get('oi_change', 0):+.2f}%")
                st.write(f"**PCR:** {data.get('pcr', 0):.2f}")

    @staticmethod
    def show_risk_analysis_popup(risk_data: Dict, title: str = "Risk Analysis"):
        """Show risk analysis in detailed popup"""
        with st.expander(f"⚠️ {title}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Portfolio Risk**")
                st.write(f"**Total Exposure:** ₹{risk_data.get('total_exposure', 0):,.2f}")
                st.write(f"**Available Capital:** ₹{risk_data.get('available_capital', 0):,.2f}")
                st.write(f"**Daily P&L:** ₹{risk_data.get('daily_pnl', 0):+,.2f}")
                st.write(f"**Risk Utilization:** {risk_data.get('risk_utilization', 0):.1f}%")

            with col2:
                st.markdown("**Risk Limits**")
                st.write(f"**Max Daily Loss:** ₹{risk_data.get('max_daily_loss', 0):,.2f}")
                st.write(f"**Position Limit:** ₹{risk_data.get('position_limit', 0):,.2f}")
                st.write(f"**Open Positions:** {risk_data.get('open_positions', 0)}")
                st.write(f"**Max Positions:** {risk_data.get('max_positions', 0)}")

            # Risk alerts
            if risk_data.get('alerts'):
                st.markdown("**Risk Alerts**")
                for alert in risk_data['alerts']:
                    if alert['level'] == 'high':
                        st.error(f"🚨 {alert['message']}")
                    elif alert['level'] == 'medium':
                        st.warning(f"⚠️ {alert['message']}")
                    else:
                        st.info(f"ℹ️ {alert['message']}")

    @staticmethod
    def create_interactive_button(label: str, action_type: str, data: Dict, key: str = None):
        """Create interactive button that shows popup on click"""
        if st.button(label, key=key, help=f"Click to view detailed {action_type}"):
            if action_type == "signal":
                PopupManager.show_signal_details(data, f"{label} Details")
            elif action_type == "position":
                PopupManager.show_position_details(data, f"{label} Details")
            elif action_type == "strategy":
                PopupManager.show_strategy_details(data, f"{label} Details")
            elif action_type == "model":
                PopupManager.show_ml_model_details(data, f"{label} Details")
            elif action_type == "live_data":
                PopupManager.show_live_data_popup(data, f"{label} Details")
            elif action_type == "risk":
                PopupManager.show_risk_analysis_popup(data, f"{label} Details")

    @staticmethod
    def show_quick_stats_modal(stats: Dict):
        """Show quick stats in a modal-like container"""
        with st.container():
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Quick Statistics</h3>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total P&L", f"₹{stats.get('total_pnl', 0):,.2f}",
                         f"{stats.get('pnl_change', 0):+.2f}%")

            with col2:
                st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%",
                         f"{stats.get('success_change', 0):+.1f}%")

            with col3:
                st.metric("Active Positions", f"{stats.get('active_positions', 0)}",
                         f"{stats.get('position_change', 0):+d}")

            with col4:
                st.metric("Capital Used", f"₹{stats.get('capital_used', 0):,.2f}",
                         f"{stats.get('capital_utilization', 0):.1f}%")

    @staticmethod
    def show_market_overview_popup():
        """Show comprehensive market overview"""
        with st.expander("🌍 Market Overview", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Index Performance**")
                indices = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"]
                for idx in indices:
                    change = f"{(0.5 - 1.5) if idx == 'BANKNIFTY' else (0.2 - 0.8):+.2f}%"
                    st.write(f"**{idx}:** {change}")

            with col2:
                st.markdown("**Market Sentiment**")
                st.write("**VIX:** 15.2 (-2.1%)")
                st.write("**PCR:** 1.25")
                st.write("**FII Flow:** ₹-125 Cr")
                st.write("**DII Flow:** ₹+340 Cr")

            with col3:
                st.markdown("**Trading Activity**")
                st.write("**Volume:** High")
                st.write("**Volatility:** Medium")
                st.write("**Trend:** Bullish")
                st.write("**Support:** 23,450")