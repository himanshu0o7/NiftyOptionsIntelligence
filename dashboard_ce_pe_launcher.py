import streamlit as st
import threading
from ce_pe_trigger_bot import CEPETriggerBot

st.set_page_config(page_title="Live CE/PE Bot Trigger", layout="centered")
st.title("🚀 Live CE/PE Trigger Bot Dashboard")

if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False

bot = CEPETriggerBot()

def launch_bot():
    if not st.session_state.bot_running:
        thread = threading.Thread(target=run_bot)
        thread.daemon = True
        thread.start()
        st.session_state.bot_running = True

def run_bot():
    bot.start_stream()
    bot.run_loop()

if not st.session_state.bot_running:
    if st.button("▶️ Start CE/PE Trigger Bot"):
        launch_bot()
        st.success("Bot is running... check terminal/logs for alerts")
else:
    st.warning("Bot is already running. Check console output for triggers.")