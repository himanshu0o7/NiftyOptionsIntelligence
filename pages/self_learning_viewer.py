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
import json
import logging
from typing import Any, Optional

import streamlit as st

from telegram_alerts import send_telegram_alert

MODULE = "self_learning_viewer"
logger = logging.getLogger(MODULE)

st.set_page_config(page_title="Self-Learning Viewer", layout="wide")


def load_evolve_data(path: str = "evolve_log.json") -> Optional[dict[str, Any] | list[dict[str, Any]]]:
    """Load evolution log data from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"{MODULE}: {path} not found")
        send_telegram_alert(f"{MODULE}: {path} not found")
    except PermissionError as exc:
        logger.error(f"{MODULE}: Permission denied for {path}: {exc}")
        send_telegram_alert(f"{MODULE}: Permission denied for {path}: {exc}")
    except json.JSONDecodeError as exc:
        logger.error(f"{MODULE}: Failed to decode JSON in {path}: {exc}")
        send_telegram_alert(f"{MODULE}: Failed to decode JSON in {path}: {exc}")
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception(f"{MODULE}: Failed to load {path}: {exc}")
        send_telegram_alert(f"{MODULE}: Failed to load {path}: {exc}")
    return None


def main() -> None:
    """Render the Self-Learning Viewer page."""
    st.title("🤖 Self-Learning Viewer")
    try:
        data = load_evolve_data()
        if data is not None:
            st.subheader("Evolution Summary")
            if isinstance(data, list):
                st.dataframe(data)
            else:
                st.json(data)
        else:
            st.warning("No evolution data available.")
    except (ValueError, TypeError, StreamlitAPIException) as exc:
        logger.exception(f"{MODULE}: Display error: {exc}")
        send_telegram_alert(f"{MODULE}: Display error - {exc}")
        st.error("Failed to display evolution data.")


if __name__ == "__main__":
    main()
