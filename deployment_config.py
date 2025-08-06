#!/usr/bin/env python3
"""
Deployment Configuration for Cloud Run
Single service on port 5000 only
"""

import os
import sys

def configure_for_deployment():
    """Configure app for Cloud Run deployment"""

    # Set environment variables for single service deployment
    os.environ['STREAMLIT_SERVER_PORT'] = '5000'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

    # Disable any ML Bot GUI components
    os.environ['DISABLE_ML_BOT_GUI'] = 'true'

    print("✅ Configured for single-service Cloud Run deployment")
    print("✅ Port: 5000")
    print("✅ ML Bot GUI disabled for deployment")

if __name__ == "__main__":
    configure_for_deployment()