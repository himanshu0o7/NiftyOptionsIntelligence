"""
Debug wrapper around the existing `strategy_config` Streamlit page.

Streamlit will silently render a blank page if exceptions occur during
module import or execution. This module provides a wrapper that
dynamically loads the original `pages/strategy_config.py` file and
exposes a `show_strategy_config` function that catches and displays
any errors using Streamlit's built‑in exception handling. Use this
wrapper while debugging to understand why the page fails to load.
"""

import importlib.util
import pathlib
import traceback
import streamlit as st


def show_strategy_config() -> None:
    """Load and execute the original strategy configuration page with error reporting."""
    # Determine the path to the original file relative to this file.
    original_path = pathlib.Path(__file__).with_name("strategy_config.py")

    try:
        # Dynamically import the original module from its file path.
        spec = importlib.util.spec_from_file_location(
            "original_strategy_config", original_path
        )
        if spec and spec.loader:
            original_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(original_module)
            # Invoke the original page function
            if hasattr(original_module, "show_strategy_config"):
                original_module.show_strategy_config()
            else:
                st.error(
                    "⚠️ The original strategy_config module does not define show_strategy_config()."
                )
        else:
            st.error(
                f"⚠️ Could not load the original strategy_config.py from {original_path}."
            )
    except Exception as exc:
        # Display the full traceback for any exception raised during import or execution
        st.error("⚠️ Error loading Strategy Configuration page")
        st.write("Below is the full traceback to help you debug:")
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        st.text(tb_str)
