import streamlit as st
from option_stream_ui import get_option_data
from ce_pe_trigger_bot import CEPETriggerBot

st.set_page_config(page_title="NiftyOptionsIntelligence", layout="wide")

st.title("📊 NiftyOptionsIntelligence - Live Market Dashboard")

st.markdown("""
Welcome to your automated Nifty options intelligence dashboard. 
Live CE/PE data + real-time bot triggers for breakout alerts.
""")

with st.sidebar:
    st.header("🔘 Navigation")
    nav = st.radio("Choose View:", ["Live Option Data", "Trigger Bot"])

if nav == "Live Option Data":
    st.subheader("📈 Live CE/PE Option Stream")
    df = get_option_data()
    st.dataframe(df, use_container_width=True)

elif nav == "Trigger Bot":
    st.subheader("🤖 Start CE/PE Breakout Bot")
    if 'bot_running' not in st.session_state:
        st.session_state.bot_running = False

    bot = CEPETriggerBot()

    def launch_bot():
        if not st.session_state.bot_running:
            thread = threading.Thread(target=run_bot)
            thread.daemon = True
            thread.start()
            st.session_state.bot_running = True

    def run