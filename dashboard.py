# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from streamlit_bokeh import streamlit_bokeh
from streamlit_analytics import track
from datetime import datetime, timedelta

# Page theme
st.set_page_config(page_title="Nifty Bot Dashboard", layout="wide")

st.markdown("""<style>
h1 { font-weight: 700; }
</style>""", unsafe_allow_html=True)

# Strategy selector
strategy = st.selectbox("Select Strategy", ["Trend Detection", "Greeks Monitor", "OI Breakout"])

# Simulate data for chart
ts = pd.date_range(datetime.now() - timedelta(minutes=60), periods=60, freq="T")
price = np.cumsum(np.random.randn(60)) + 20000
greeks = np.random.rand(60)*10 - 5

# Show analytics tracking
with track(page_name="dashboard", enable=True):
    st.header(f"{strategy} Module")

    # Price trend chart
    p = figure(title="Price Trend", x_axis_type="datetime", plot_width=800, plot_height=300)
    p.line(ts, price, color="blue", legend_label="Price")
    streamlit_bokeh(p, use_container_width=True)

    # Greek display
    st.metric("Delta", f"{greeks[-1]:.2f}", delta=f"{greeks[-1] - greeks[-2]:.2f}")

    # Strategy logic placeholder
    st.write("Next expected signal:", "📈" if greeks[-1] > 0 else "📉")

    # Footer timestamp
    st.caption(f"Updated at {datetime.now().strftime('%H:%M:%S')}")

# Toggle analytics via URL
st.write("To view widget analytics, add '?analytics=on' to the app URL")

