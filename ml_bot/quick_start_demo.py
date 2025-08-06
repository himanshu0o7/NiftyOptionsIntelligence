#!/usr/bin/env python3
"""
Quick Start Demo: How to use ML Bot features
"""

import streamlit as st
import json
from datetime import datetime

def main():
    st.title("🤖 ML Bot Quick Start Demo")

    st.markdown("""
    ## How to Use the ML Bot System

    **Two Applications Running:**
    - **Main Trading**: http://localhost:5000 (Complete trading dashboard)
    - **ML Bot GUI**: http://localhost:8501 (AI evolution features)
    """)

    # Feature demonstration
    tab1, tab2, tab3 = st.tabs(["🧠 AI Strategy Generator", "🔧 Auto Error Fixing", "📊 Performance Analysis"])

    with tab1:
        st.header("AI Strategy Generator Demo")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Parameters")
            market = st.selectbox("Market Condition", ["Bullish", "Bearish", "Sideways"])
            index = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
            risk = st.slider("Risk Level", 1, 10, 5)

        with col2:
            st.subheader("Generated Strategy")
            if st.button("🚀 Generate AI Strategy"):
                with st.spinner("GPT-4o analyzing market conditions..."):
                    strategy = {
                        "name": f"{market} {index} AI Strategy",
                        "entry_condition": f"Market trend: {market}, Sentiment > 0.6",
                        "strike_selection": "ATM CE for bullish, ATM PE for bearish",
                        "position_size": f"₹{min(3400, risk * 300)}",
                        "stop_loss": f"{max(2, 8-risk)}%",
                        "take_profit": f"{min(15, 8+risk)}%",
                        "expected_accuracy": f"{65 + risk}%",
                        "ai_reasoning": f"Based on {market.lower()} sentiment and {index} technical analysis"
                    }

                    st.success("✅ Strategy Generated!")
                    st.json(strategy)

    with tab2:
        st.header("Auto Error Fixing Demo")

        st.info("**Error Detection & Auto-Fix System**")

        if st.button("🔍 Scan for Errors"):
            with st.spinner("AI scanning system for issues..."):
                errors = {
                    "critical_errors": 0,
                    "warnings": 1,
                    "performance_issues": 1,
                    "auto_fixes_available": 2
                }

                st.success("✅ System Scan Complete!")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Critical Errors", errors["critical_errors"], delta="✅ None")
                with col2:
                    st.metric("Warnings", errors["warnings"], delta="⚠️ Minor")
                with col3:
                    st.metric("Performance Issues", errors["performance_issues"], delta="📊 Detected")
                with col4:
                    st.metric("Auto-Fixes", errors["auto_fixes_available"], delta="🔧 Ready")

                # Show sample auto-fix
                with st.expander("🔧 Sample Auto-Fix"):
                    st.code("""
# Error: Feature dimension mismatch
# Auto-generated fix:

def validate_features(X_train, expected_features=10):
    if X_train.shape[1] != expected_features:
        # Auto-fix: Adjust feature selection
        X_train = X_train.iloc[:, :expected_features]
        logger.info(f"Features adjusted to {expected_features}")
    return X_train
                    """)
                    st.success("✅ Auto-fix applied successfully!")

    with tab3:
        st.header("Performance Analysis Demo")

        st.info("**AI-Powered Performance Analysis**")

        if st.button("📊 Analyze Performance"):
            with st.spinner("GPT-4o analyzing trading performance..."):
                analysis = {
                    "overall_score": "B+ (Good Performance)",
                    "accuracy": "73.2% (Above average)",
                    "key_strengths": [
                        "Strong breakout strategy performance",
                        "Good risk management",
                        "Consistent profit factors"
                    ],
                    "improvement_areas": [
                        "ML signal accuracy needs improvement",
                        "Position sizing could be optimized",
                        "Consider volatility-based adjustments"
                    ],
                    "ai_recommendations": [
                        "Increase Random Forest n_estimators to 200",
                        "Add VIX-based position sizing",
                        "Implement dynamic stop-loss based on Greeks"
                    ]
                }

                st.success("✅ AI Analysis Complete!")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("✅ Strengths")
                    for strength in analysis["key_strengths"]:
                        st.write(f"• {strength}")

                with col2:
                    st.subheader("🎯 Improvements")
                    for improvement in analysis["improvement_areas"]:
                        st.write(f"• {improvement}")

                st.subheader("🧠 AI Recommendations")
                for rec in analysis["ai_recommendations"]:
                    st.write(f"• {rec}")

    # Quick access guide
    st.markdown("---")
    st.markdown("""
    ## 🚀 Quick Access Guide

    **Main Trading System (Port 5000):**
    - Live trading dashboard
    - Multi-index support (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, NIFTYNXT50)
    - Real-time signals and order placement
    - Risk management and P&L tracking

    **ML Bot GUI (Port 8501):**
    - AI-powered strategy generation
    - Automatic error detection and fixing
    - Performance analysis with GPT-4o
    - Continuous learning and evolution

    **Key Benefits:**
    ✓ Automated trading with AI enhancement
    ✓ Self-improving algorithms
    ✓ Professional risk management
    ✓ Multi-index support with proper Greeks validation
    """)

if __name__ == "__main__":
    main()