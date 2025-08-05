import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import yaml

from risk_management.risk_calculator import RiskCalculator
from risk_management.position_manager import PositionManager
from telegram_alerts import send_telegram_alert


def show_risk_monitor():
    """Display comprehensive risk monitoring dashboard"""

    st.header("🛡️ Risk Management Monitor")

    # Determine default config path relative to this file
    default_cfg_path = os.path.join(os.path.dirname(__file__), "risk_config.yaml")
    cfg_path = st.sidebar.text_input("Risk config file path", value=default_cfg_path)

    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh).get("risk_management", {})

        risk_calc = RiskCalculator(cfg)
        pos_manager = PositionManager(cfg)

        positions_df = pos_manager.get_positions_dataframe()
        returns = compute_portfolio_returns(positions_df, risk_calc)
        metrics = get_portfolio_risk_metrics(risk_calc, positions_df, returns)
        exposure = (positions_df["Quantity"].abs() * positions_df["Current Price"]).sum()
        margin_used = exposure * 0.1

        show_risk_overview(metrics, exposure, margin_used, risk_calc)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["🎯 Position Risk", "📊 Portfolio Risk", "⚠️ Risk Alerts", "🧪 Stress Testing"]
        )

        with tab1:
            show_position_risk(pos_manager)

        with tab2:
            show_portfolio_risk(risk_calc, pos_manager, metrics, returns, exposure)

        with tab3:
            show_risk_alerts(risk_calc, pos_manager, metrics, exposure)

        with tab4:
            show_stress_testing(risk_calc, pos_manager)

    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] Initialization error: {exc}")
        st.error("Failed to load risk monitor")


def show_risk_overview(metrics, exposure, margin_used, risk_calc):
    """Display high-level risk metrics"""
    try:
        st.subheader("📈 Risk Overview")
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.metric(
                "Portfolio VaR (95%)", f"₹{metrics['var_95'] * exposure:,.0f}"
            )

        with c2:
            st.metric("Exposure", f"₹{exposure:,.0f}")

        with c3:
            st.metric("Margin Used", f"₹{margin_used:,.0f}")

        with c4:
            st.metric("Portfolio Beta", f"{metrics['beta']:.2f}")

        status = get_current_risk_status(metrics, risk_calc, exposure)
        with c5:
            st.metric("Risk Status", status)

        if status == "Low":
            st.success(
                "🟢 **Risk Status: LOW** - Portfolio is within acceptable limits"
            )
        elif status == "Medium":
            st.warning(
                "🟡 **Risk Status: MEDIUM** - Monitor closely for limit breaches"
            )
        else:
            st.error(
                "🔴 **Risk Status: HIGH** - Immediate attention required!"
            )
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] show_risk_overview error: {exc}")
        st.error("Unable to display risk overview")


def show_position_risk(pos_manager: PositionManager):
    """Display individual position risk analysis"""
    try:
        st.subheader("🎯 Position-Level Risk Analysis")
        col1, col2 = st.columns([1, 3])

        with col1:
            st.selectbox(
                "Sort Positions By",
                ["Risk Score", "P&L", "Position Size", "Time to Expiry"],
            )
            st.selectbox(
                "Risk Filter",
                ["All Positions", "High Risk", "Medium Risk", "Low Risk"],
            )
            show_greeks = st.checkbox("Show Greeks", value=True)

        with col2:
            positions_df = get_positions_risk_data(pos_manager)
            if not positions_df.empty:
                styled = style_position_risk_table(positions_df)
                st.dataframe(styled, use_container_width=True, height=400)

                high_risk = positions_df[positions_df["Risk_Score"] == "High"]
                if not high_risk.empty:
                    st.warning(
                        f"⚠️ {len(high_risk)} positions flagged as HIGH RISK"
                    )
                    for _, row in high_risk.iterrows():
                        st.error(f"🚨 {row['Symbol']}: {row['Risk_Reason']}")
            else:
                st.info("No active positions to analyze")

        if show_greeks:
            show_position_greeks_analysis(pos_manager)
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] show_position_risk error: {exc}")
        st.error("Unable to display position risk")


def show_portfolio_risk(risk_calc, pos_manager, metrics, returns, exposure):
    """Display portfolio-level risk metrics"""
    try:
        st.subheader("📊 Portfolio Risk Analytics")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Risk Metrics**")
            metrics_df = pd.DataFrame(
                [
                    {
                        "Metric": "Value at Risk (95%)",
                        "Value": f"₹{metrics['var_95'] * exposure:,.0f}",
                        "Limit": f"₹{risk_calc.max_var_limit:,.0f}",
                    },
                    {
                        "Metric": "Value at Risk (99%)",
                        "Value": f"₹{metrics['var_99'] * exposure:,.0f}",
                        "Limit": f"₹{risk_calc.max_var_limit:,.0f}",
                    },
                    {
                        "Metric": "Expected Shortfall",
                        "Value": f"₹{metrics['expected_shortfall'] * exposure:,.0f}",
                        "Limit": f"₹{risk_calc.max_var_limit:,.0f}",
                    },
                    {
                        "Metric": "Maximum Drawdown",
                        "Value": f"{metrics['max_drawdown'] * 100:.1f}%",
                        "Limit": "15%",
                    },
                    {
                        "Metric": "Portfolio Beta",
                        "Value": f"{metrics['beta']:.2f}",
                        "Limit": f"{risk_calc.max_portfolio_beta}",
                    },
                    {
                        "Metric": "Correlation to Market",
                        "Value": f"{metrics['correlation']:.2f}",
                        "Limit": f"{risk_calc.max_correlation}",
                    },
                ]
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        with col2:
            st.write("**Risk Distribution**")
            rc_data = get_risk_contribution_data(risk_calc, pos_manager)
            fig_risk = px.pie(
                rc_data,
                values="Risk_Contribution",
                names="Position",
                title="Risk Contribution by Position",
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        st.write("**Portfolio Greeks**")
        g1, g2, g3, g4, g5 = st.columns(5)
        greeks = get_portfolio_greeks(pos_manager)

        with g1:
            color = "normal" if abs(greeks['delta']) < 50 else "inverse"
            st.metric("Total Delta", f"{greeks['delta']:.1f}", delta_color=color)
        with g2:
            st.metric("Total Gamma", f"{greeks['gamma']:.3f}")
        with g3:
            st.metric("Total Theta", f"{greeks['theta']:.0f}")
        with g4:
            st.metric("Total Vega", f"{greeks['vega']:.1f}")
        with g5:
            st.metric("Total Rho", f"{greeks['rho']:.2f}")

        st.write("**Historical Risk Trend**")
        hist_chart = create_historical_risk_chart(returns, risk_calc)
        st.plotly_chart(hist_chart, use_container_width=True)
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] show_portfolio_risk error: {exc}")
        st.error("Unable to display portfolio risk")


def show_risk_alerts(risk_calc, pos_manager, metrics, exposure):
    """Display risk alerts and warnings"""
    try:
        st.subheader("⚠️ Risk Alerts & Warnings")

        alerts = get_current_risk_alerts(risk_calc, pos_manager, metrics, exposure)
        if alerts:
            st.write("**🚨 Active Alerts**")
            for alert in alerts:
                if alert["severity"] == "Critical":
                    st.error(f"🔴 **{alert['type']}**: {alert['message']}")
                elif alert["severity"] == "High":
                    st.warning(f"🟡 **{alert['type']}**: {alert['message']}")
                else:
                    st.info(f"🔵 **{alert['type']}**: {alert['message']}")
        else:
            st.success("✅ No active risk alerts")

        st.divider()

        st.write("**Risk Limits Status**")
        limits_df = get_risk_limits_status(risk_calc, metrics, exposure)
        st.dataframe(limits_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] show_risk_alerts error: {exc}")
        st.error("Unable to display risk alerts")


def show_stress_testing(risk_calc, pos_manager):
    """Display stress testing scenarios and results"""
    try:
        st.subheader("🧪 Stress Testing & Scenario Analysis")
        col1, col2 = st.columns([1, 2])

        with col1:
            scenario_type = st.selectbox(
                "Stress Test Scenario",
                [
                    "Market Crash (-20%)",
                    "Moderate Decline (-10%)",
                    "Volatility Spike (+50%)",
                    "Interest Rate Shock (+2%)",
                    "Custom Scenario",
                ],
            )

            if scenario_type == "Custom Scenario":
                market_shock = st.slider("Market Movement (%)", -30, 30, 0, 1)
                vol_shock = st.slider("Volatility Change (%)", -50, 100, 0, 5)
                ir_shock = st.slider("Interest Rate Change (%)", -2, 5, 0, 0.25)
            else:
                market_shock, vol_shock, ir_shock = get_predefined_scenario(
                    scenario_type
                )

            if st.button("🧪 Run Stress Test", type="primary"):
                run_stress_test(risk_calc, pos_manager, market_shock, vol_shock, ir_shock)

        with col2:
            st.write("**Stress Test Results**")
            results = st.session_state.get("stress_test_results")
            if results:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Portfolio Impact", f"₹{results['total_impact']:,.0f}")
                with c2:
                    pct = (
                        results['total_impact'] / results['current_value'] * 100
                        if results['current_value'] else 0
                    )
                    st.metric("Impact %", f"{pct:.1f}%")
                with c3:
                    status = "✅ Gain" if results['total_impact'] > 0 else "⚠️ Loss"
                    st.metric("Risk Status", status)

                impact_df = pd.DataFrame(results['position_impacts'])
                fig_imp = px.bar(
                    impact_df,
                    x="Symbol",
                    y="Impact",
                    color="Impact",
                    color_continuous_scale="RdYlGn",
                    title="Stress Test Impact by Position",
                )
                st.plotly_chart(fig_imp, use_container_width=True)
                st.dataframe(impact_df, use_container_width=True, hide_index=True)
            else:
                st.info("Run a stress test to see results")
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] show_stress_testing error: {exc}")
        st.error("Unable to display stress testing")


# Helper functions ---------------------------------------------------------


def get_current_risk_status(metrics, risk_calc, exposure) -> str:
    """Determine current overall risk status"""
    try:
        var_amt = metrics["var_95"] * exposure
        if var_amt < risk_calc.max_var_limit * 0.5:
            return "Low"
        if var_amt < risk_calc.max_var_limit:
            return "Medium"
        return "High"
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] get_current_risk_status error: {exc}")
        return "Unknown"


def get_positions_risk_data(pos_manager: PositionManager) -> pd.DataFrame:
    """Get position-level risk data"""
    try:
        df = pos_manager.get_positions_dataframe()
        if df.empty:
            return df
        df["Position_Size"] = df["Quantity"] * df["Current Price"]
        df["Risk_Score"] = np.where(df["P&L %"].abs() > 10, "High", "Low")
        df["Risk_Reason"] = np.where(
            df["Risk_Score"] == "High", "Large P&L swing", "Within limits"
        )
        return df
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] get_positions_risk_data error: {exc}")
        return pd.DataFrame()


def style_position_risk_table(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for styling - returns dataframe as-is"""
    return df


def show_position_greeks_analysis(pos_manager: PositionManager):
    """Show detailed Greeks analysis for positions"""
    try:
        st.write("**🏛️ Position Greeks Analysis**")
        gdf = get_position_greeks_data(pos_manager)
        if gdf.empty:
            st.info("No Greeks data available")
            return
        c1, c2 = st.columns(2)
        with c1:
            fig_d = px.bar(
                gdf, x="Symbol", y="Delta", title="Delta Exposure by Position", color="Delta"
            )
            st.plotly_chart(fig_d, use_container_width=True)
        with c2:
            fig_t = px.bar(
                gdf, x="Symbol", y="Theta", title="Theta Decay by Position", color="Theta"
            )
            st.plotly_chart(fig_t, use_container_width=True)
    except Exception as exc:
        send_telegram_alert(
            f"[RiskMonitor] show_position_greeks_analysis error: {exc}"
        )
        st.error("Unable to display Greeks analysis")


def get_position_greeks_data(pos_manager: PositionManager) -> pd.DataFrame:
    try:
        data = []
        for pos in pos_manager.active_positions.values():
            if "CE" in pos.symbol or "PE" in pos.symbol:
                strike = pos_manager._extract_strike_from_symbol(pos.symbol)
                expiry = pos_manager._extract_expiry_from_symbol(pos.symbol)
                opt_type = "CE" if "CE" in pos.symbol else "PE"
                if strike and expiry:
                    greeks = pos_manager.greeks_calc.calculate_all_greeks(
                        pos.current_price * 100, strike, expiry, opt_type
                    )
                    data.append(
                        {
                            "Symbol": pos.symbol,
                            "Delta": greeks.get("delta", 0) * pos.quantity,
                            "Gamma": greeks.get("gamma", 0) * pos.quantity,
                            "Theta": greeks.get("theta", 0) * pos.quantity,
                            "Vega": greeks.get("vega", 0) * pos.quantity,
                        }
                    )
        return pd.DataFrame(data)
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] get_position_greeks_data error: {exc}")
        return pd.DataFrame()


def compute_portfolio_returns(df: pd.DataFrame, risk_calc: RiskCalculator) -> pd.Series:
    """Aggregate position returns into portfolio returns"""
    try:
        if df.empty:
            return pd.Series(dtype=float)
        total_val = (df["Quantity"].abs() * df["Current Price"]).sum()
        returns_dict = {}
        for _, row in df.iterrows():
            ret = risk_calc._get_symbol_returns(row["Symbol"])
            if ret.empty or total_val == 0:
                continue
            weight = abs(row["Quantity"] * row["Current Price"]) / total_val
            returns_dict[row["Symbol"]] = ret * weight
        if not returns_dict:
            return pd.Series(dtype=float)
        returns_df = pd.DataFrame(returns_dict)
        return returns_df.sum(axis=1)
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] compute_portfolio_returns error: {exc}")
        return pd.Series(dtype=float)


def get_portfolio_risk_metrics(risk_calc, df, returns) -> dict:
    try:
        if returns.empty:
            return {
                "var_95": 0,
                "var_99": 0,
                "expected_shortfall": 0,
                "max_drawdown": 0,
                "beta": 0,
                "correlation": 0,
            }
        var95 = risk_calc.calculate_var(returns, 0.95)
        var99 = risk_calc.calculate_var(returns, 0.99)
        es = risk_calc.calculate_expected_shortfall(returns, 0.95)
        md, _, _ = risk_calc.calculate_maximum_drawdown(returns)
        market_returns = risk_calc._get_market_returns()
        beta = risk_calc.calculate_beta(returns, market_returns)
        corr = risk_calc.calculate_correlation(returns, market_returns)
        return {
            "var_95": var95,
            "var_99": var99,
            "expected_shortfall": es,
            "max_drawdown": md,
            "beta": beta,
            "correlation": corr,
        }
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] get_portfolio_risk_metrics error: {exc}")
        return {
            "var_95": 0,
            "var_99": 0,
            "expected_shortfall": 0,
            "max_drawdown": 0,
            "beta": 0,
            "correlation": 0,
        }


def get_risk_contribution_data(risk_calc, pos_manager) -> pd.DataFrame:
    try:
        positions = []
        for pos in pos_manager.active_positions.values():
            returns = risk_calc._get_symbol_returns(pos.symbol)
            volatility = returns.std() if not returns.empty else 0
            positions.append(
                {
                    "symbol": pos.symbol,
                    "current_value": abs(pos.quantity * pos.current_price),
                    "volatility": volatility,
                }
            )
        contrib = risk_calc.calculate_risk_contribution(positions)
        if not contrib:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {"Position": sym, "Risk_Contribution": val * 100}
                for sym, val in contrib.items()
            ]
        )
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] get_risk_contribution_data error: {exc}")
        return pd.DataFrame()


def get_portfolio_greeks(pos_manager) -> dict:
    try:
        summary = pos_manager.get_portfolio_summary()
        g = summary.get("portfolio_greeks", {})
        return {
            "delta": g.get("total_delta", 0),
            "gamma": g.get("total_gamma", 0),
            "theta": g.get("total_theta", 0),
            "vega": g.get("total_vega", 0),
            "rho": g.get("total_rho", 0),
        }
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] get_portfolio_greeks error: {exc}")
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}


def create_historical_risk_chart(returns: pd.Series, risk_calc) -> go.Figure:
    try:
        if returns.empty:
            return go.Figure()
        rolling_var = (
            returns.rolling(window=30)
            .apply(lambda x: risk_calc.calculate_var(pd.Series(x), 0.95))
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling_var.index, y=rolling_var.values, mode="lines", name="VaR (95%)"))
        fig.add_hline(y=risk_calc.max_var_limit, line_dash="dash", line_color="orange", annotation_text="VaR Limit")
        fig.update_layout(title="Historical VaR Trend", xaxis_title="Date", yaxis_title="VaR")
        return fig
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] create_historical_risk_chart error: {exc}")
        return go.Figure()


def get_current_risk_alerts(risk_calc, pos_manager, metrics, exposure) -> list:
    try:
        alerts = []
        warnings = pos_manager.get_risk_warnings()
        for w in warnings:
            alerts.append({"type": "Position", "severity": "High", "message": w})
        violations = risk_calc.check_risk_limits(metrics, exposure)
        for v in violations:
            alerts.append({"type": "Limit", "severity": "Critical", "message": v})
        return alerts
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] get_current_risk_alerts error: {exc}")
        return []


def get_risk_limits_status(risk_calc, metrics, exposure) -> pd.DataFrame:
    try:
        data = [
            {
                "Metric": "Portfolio VaR",
                "Current": metrics["var_95"] * exposure,
                "Limit": risk_calc.max_var_limit,
                "Status": "OK"
                if metrics["var_95"] * exposure <= risk_calc.max_var_limit
                else "Breach",
            },
            {
                "Metric": "Portfolio Beta",
                "Current": metrics["beta"],
                "Limit": risk_calc.max_portfolio_beta,
                "Status": "OK"
                if abs(metrics["beta"]) <= risk_calc.max_portfolio_beta
                else "Breach",
            },
            {
                "Metric": "Correlation",
                "Current": metrics["correlation"],
                "Limit": risk_calc.max_correlation,
                "Status": "OK"
                if abs(metrics["correlation"]) <= risk_calc.max_correlation
                else "Breach",
            },
        ]
        return pd.DataFrame(data)
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] get_risk_limits_status error: {exc}")
        return pd.DataFrame()


def get_predefined_scenario(name: str):
    scenarios = {
        "Market Crash (-20%)": (-20, 50, 0),
        "Moderate Decline (-10%)": (-10, 25, 0),
        "Volatility Spike (+50%)": (0, 50, 0),
        "Interest Rate Shock (+2%)": (0, 10, 2),
    }
    return scenarios.get(name, (0, 0, 0))


def run_stress_test(risk_calc, pos_manager, market_shock, vol_shock, ir_shock):
    try:
        impacts = []
        total = 0
        for pos in pos_manager.active_positions.values():
            greeks = get_position_greeks_data(pos_manager)
            g = greeks[greeks["Symbol"] == pos.symbol]
            if g.empty:
                continue
            delta = g.iloc[0]["Delta"]
            gamma = g.iloc[0]["Gamma"]
            vega = g.iloc[0]["Vega"]
            current_val = abs(pos.quantity * pos.current_price)
            d_pnl = delta * current_val * (market_shock / 100)
            g_pnl = 0.5 * gamma * current_val * ((market_shock / 100) ** 2)
            v_pnl = vega * (vol_shock / 100)
            impact = d_pnl + g_pnl + v_pnl
            total += impact
            impacts.append(
                {
                    "Symbol": pos.symbol,
                    "Current_Value": current_val,
                    "Impact": impact,
                }
            )

        st.session_state.stress_test_results = {
            "scenario": f"Market: {market_shock}%, Vol: {vol_shock}%, IR: {ir_shock}%",
            "current_value": sum(
                abs(p.quantity * p.current_price)
                for p in pos_manager.active_positions.values()
            ),
            "total_impact": total,
            "position_impacts": impacts,
        }
        st.rerun()
    except Exception as exc:
        send_telegram_alert(f"[RiskMonitor] run_stress_test error: {exc}")


def get_scenario_history() -> list:
    return []


