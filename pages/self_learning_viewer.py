import streamlit as st
import json
import os
from datetime import datetime

# ─────────────────────────────────────────────
# ✅ Setup Page
# ─────────────────────────────────────────────
st.set_page_config(page_title="📘 AI Self-Learning Log", layout="wide")
st.title("🤖 Self-Learning Log Viewer")
st.markdown("Displays what KP5Bot has learned from the web (Trendlyne, Google News, YouTube, etc.)")

log_path = "logs/self_learning.jsonl"
if not os.path.exists(log_path):
    st.warning("No learning logs found.")
    st.stop()

# ─────────────────────────────────────────────
# ✅ Load Logs
# ─────────────────────────────────────────────
with open(log_path, "r") as f:
    logs = [json.loads(line.strip()) for line in f if line.strip()]

# ─────────────────────────────────────────────
# ✅ Sidebar: Topic Filter
# ─────────────────────────────────────────────
topics = sorted(set(log["topic"] for log in logs))
selected_topic = st.sidebar.selectbox("🔍 Select Topic", topics)

filtered_logs = [log for log in logs if log["topic"] == selected_topic]

# ─────────────────────────────────────────────
# ✅ Display Logs
# ─────────────────────────────────────────────
for log in filtered_logs[::-1]:  # Newest first
    st.markdown(f"### 📅 {log['timestamp']}")
    for item in log["results"]:
        st.markdown(f"**🔗 URL:** [{item['url']}]({item['url']})")
        st.markdown(f"""
        **📝 Summary:**

        {item['summary']}
        """)
        st.markdown("---")
