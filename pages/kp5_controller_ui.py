"""Streamlit UI hooks for KP5Controller."""

from __future__ import annotations

import streamlit as st

from kp5_controller import run_self_audit, run_web_learner

st.title("KP5 Controller")

if st.button("Run Self Audit"):
    try:
        result = run_self_audit()
        st.success(result)
    except Exception as exc:  # pragma: no cover - UI only
        st.error(f"Self audit failed: {exc}")

if st.button("Run Web Learner"):
    try:
        result = run_web_learner()
        st.success(result)
    except Exception as exc:  # pragma: no cover - UI only
        st.error(f"Web learner failed: {exc}")
