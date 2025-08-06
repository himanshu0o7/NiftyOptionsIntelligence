import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
from datetime import datetime

# ✅ Set Page Config
st.set_page_config(page_title="📖 KP5 Self Learning Viewer", layout="wide")
st.title("📚 KP5Bot Self Learning Log Viewer")

# ✅ Load log file
log_path = "logs/memory_log.json"
if not os.path.exists(log_path):
    st.warning("No learning log found.")
    st.stop()

entries = []
with open(log_path, "r") as file:
    for line in file:
        try:
            data = json.loads(line.strip())
            entries.append(data)
        except json.JSONDecodeError:
            continue

if not entries:
    st.warning("Learning log is empty or invalid.")
    st.stop()

# ✅ Convert to DataFrame
df = pd.DataFrame(entries)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ✅ Sidebar filters
all_topics = sorted(df['topic'].unique())
selected_topics = st.sidebar.multiselect("🧠 Select Topics", all_topics, default=all_topics)
date_range = st.sidebar.date_input("📅 Date Range", [df['timestamp'].min().date(), df['timestamp'].max().date()])

# ✅ Apply filters
filtered = df[
    (df['topic'].isin(selected_topics)) &
    (df['timestamp'].dt.date >= date_range[0]) &
    (df['timestamp'].dt.date <= date_range[1])
]

# ✅ Chart - Learning Frequency
st.subheader("📊 Learning Frequency Over Time")
daily_count = filtered.groupby(filtered['timestamp'].dt.date).size().reset_index(name='Learnings')
fig = px.bar(daily_count, x='timestamp', y='Learnings', title="Daily Learning Events")
st.plotly_chart(fig, use_container_width=True)

# ✅ Detailed Comparison
st.subheader("📋 Learning Summaries by Topic")
for idx, row in filtered.iterrows():
    st.markdown(f"### 📌 {row['topic']}")
    st.caption(f"🕒 {row['timestamp']}")
    for res in row['results']:
        st.markdown(f"**🔗 URL:** [{res['url']}]({res['url']})")
        st.markdown("**📝 Summary:**")
        st.write(res['summary'])
        st.markdown("---")

# ✅ Export section
st.subheader("📦 Export Options")

# Prepare flat exportable data
export_rows = []
for idx, row in filtered.iterrows():
    for res in row['results']:
        export_rows.append({
            "timestamp": row['timestamp'],
            "topic": row['topic'],
            "url": res['url'],
            "summary": res['summary']
        })

export_df = pd.DataFrame(export_rows)
st.download_button("📁 Export CSV", export_df.to_csv(index=False), file_name="kp5bot_learning_log.csv")

# Optional: Export PDF (not native in Streamlit - needs workaround or separate script)
# e.g., use pdfkit, reportlab, or send to backend

st.success("✅ Done displaying filtered learning logs.")
