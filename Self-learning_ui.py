import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime

# Load learn logs
def load_learning_logs(log_file="logs/self_learning_log.json"):
    if not os.path.exists(log_file):
        return pd.DataFrame(columns=["timestamp", "topic", "url", "summary"])

    logs = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                for r in entry.get("results", []):
                    logs.append({
                        "timestamp": entry.get("timestamp"),
                        "topic": entry.get("topic"),
                        "url": r.get("url"),
                        "summary": r.get("summary")[:300] + "..."
                    })
            except json.JSONDecodeError:
                continue

    return pd.DataFrame(logs)

# Streamlit UI
st.set_page_config(page_title="📚 KP5Bot Learning Logs", layout="wide")
st.title("📘 KP5Bot Self Learning Log Viewer")

log_df = load_learning_logs()

if log_df.empty:
    st.warning("No learning logs found yet. Run self_learner.py to start.")
else:
    with st.expander("📊 Learning Summary Stats"):
        st.metric("Total Learning Sessions", log_df["timestamp"].nunique())
        st.metric("Unique Topics", log_df["topic"].nunique())
        st.metric("Unique URLs Fetched", log_df["url"].nunique())

    with st.expander("🧠 Learning Entries"):
        topic_filter = st.selectbox("Filter by Topic", options=["All"] + sorted(log_df["topic"].unique().tolist()))
        if topic_filter != "All":
            filtered_df = log_df[log_df["topic"] == topic_filter]
        else:
            filtered_df = log_df

        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    st.info("Next: Integrate YouTube, TV & news parsing using scraping, RSS, or offline tools.")

# Future: Add charts, learning rate tracker, auto-patch stats
