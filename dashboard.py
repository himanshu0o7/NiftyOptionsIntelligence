import streamlit as st

st.title("🧠 KP5Bot Options Dashboard")
strike = st.selectbox("Select Strike", [22500, 22600, 22700])
option_type = st.radio("Option Type", ["CE", "PE"])
if st.button("Fetch Signal"):
    st.success(f"Analyzing {strike} {option_type}... (soon integrated)")

