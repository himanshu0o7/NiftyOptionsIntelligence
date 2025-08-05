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
