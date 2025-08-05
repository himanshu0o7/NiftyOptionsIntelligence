import streamlit as st
import pandas as pd
from utils.sensibull_greeks_fetcher import fetch_option_data
from option_stream_ui import get_nifty_option_tokens

def load_and_filter_data():
    ce_df, pe_df = get_nifty_option_tokens()
    if ce_df is None or pe_df is None:
        st.warning("⚠️ Unable to load option tokens.")
        return pd.DataFrame()

    token_df = pd.concat([ce_df, pe_df])
    token_df = token_df.drop_duplicates(subset=["symbol"])

    records = []
    for _, row in token_df.iterrows():
        symbol = row["symbol"]
        token = row["token"]
        strike = row["strike"]
        expiry = row["expiry"]
        option_type = row["optiontype"]
        name = row["name"]

        data = fetch_option_data(token)
        if not data:
            continue

        delta = data.get("delta", 0)
        theta = data.get("theta", 0)
        vega = data.get("vega", 0)
        volume = data.get("volume", 0)
        oi = data.get("oi", 0)
        ltp = data.get("ltp", 0)

        if all(k is not None for k in [delta, theta, vega]):
            records.append({
                "Index": name,
                "Strike": strike,
                "Type": option_type,
                "Symbol": symbol,
                "Expiry": expiry,
                "LTP": ltp,
                "Volume": volume,
                "OI": oi,
                "Delta": round(delta, 2),
                "Theta": round(theta, 2),
                "Vega": round(vega, 2),
            })

    return pd.DataFrame(records)

def render_scanner():
    st.header("📊 Multi-Strike Option Scanner (NIFTY & BANKNIFTY)")
    df = load_and_filter_data()

    if df.empty:
        st.info("No data to display.")
        return


    greek_filter = st.sidebar.checkbox("📌 Filter: Delta > 0.4, |Theta| < 10, Vega > 1.5, Vol > 2000", value=True)
    if greek_filter:
        df = df[
            (df["Delta"].abs() > 0.4) &
            (df["Theta"].abs() < 10) &
            (df["Vega"] > 1.5) &
            (df["Volume"] > 2000)
        ]

    df = df.sort_values(by="Volume", ascending=False)
    st.dataframe(df, use_container_width=True)

def render_scanner():
    import streamlit as st
    st.warning("🛠️ Greek scanner is under construction.")
x
