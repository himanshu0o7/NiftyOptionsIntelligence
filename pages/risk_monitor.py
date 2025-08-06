"""
Risk Monitor Page

This Streamlit app page displays real-time risk monitoring based on current positions
using PositionManager and RiskCalculator components.

Author: Python Code Expert
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from risk_management.position_manager import PositionManager
from risk_management.risk_calculator import RiskCalculator

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def show_risk_monitor() -> None:
    """Main function to render the Risk Monitor page in Streamlit."""
    st.title("📊 Options Risk Monitor")
    st.write("Monitor your open positions and their associated risks.")

    try:
        # Initialize manager and calculator
        logger.debug("Initializing PositionManager")
        position_manager = PositionManager()

        logger.debug("Fetching open positions")
        open_positions = position_manager.get_open_positions()

        if open_positions.empty:
            st.info("No open positions found.")
            return

        logger.debug("Calculating risk")
        risk_calculator = RiskCalculator()
        risk_data = risk_calculator.calculate_risk(open_positions)

        logger.debug("Displaying risk data")
        display_risk_dashboard(open_positions, risk_data)

    except Exception as e:
        logger.exception("Failed to load risk monitor")
        st.error(f"❌ An error occurred while loading the risk monitor: {e}")


def display_risk_dashboard(positions: pd.DataFrame, risks: pd.DataFrame) -> None:
    """
    Display risk-related charts and metrics in Streamlit.

    Args:
        positions (pd.DataFrame): DataFrame containing position details.
        risks (pd.DataFrame): DataFrame containing calculated risk metrics.
    """
    st.subheader("📈 Open Positions Overview")
    st.dataframe(positions, use_container_width=True)

    st.subheader("⚠️ Risk Analysis Summary")
    st.dataframe(risks, use_container_width=True)

    if "Greeks" in risks.columns:
        st.subheader("Delta vs Gamma")
        fig = px.scatter(
            risks,
            x="Delta",
            y="Gamma",
            size="Exposure",
            color="Symbol",
            hover_data=["Symbol", "Delta", "Gamma", "Theta", "Vega"],
            title="Delta vs Gamma Bubble Chart",
        )
        st.plotly_chart(fig, use_container_width=True)


# Entry point for Streamlit
if __name__ == "__main__":
    show_risk_monitor()
