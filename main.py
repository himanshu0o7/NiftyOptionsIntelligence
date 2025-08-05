import streamlit as st
import threading

from option_stream_ui import get_option_data
from ce_pe_trigger_bot import CEPETriggerBot
from auto_planner import AutoPlanner
from self_audit import SelfAudit
from module_creator import ModuleCreator
from self_improver import SelfImprover
from risk_guard import RiskGuard

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

    def run_bot():
        RiskGuard.check_safety()
        goal = AutoPlanner().decide_next_goal()

        if goal == "create_module":
            missing = SelfAudit().find_missing_modules()
            ModuleCreator().generate(missing)

        elif goal == "improve_code":
            SelfImprover().run_improvement_cycle()

        RiskGuard.log_success()

    def launch_bot():
        if not st.session_state.bot_running:
            thread = threading.Thread(target=run_bot)
            thread.daemon = True
            thread.start()
            st.session_state.bot_running = True
            st.success("Bot started successfully!")

    if st.button("🚀 Launch Bot"):
        launch_bot()
