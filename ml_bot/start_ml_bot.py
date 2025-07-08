#!/usr/bin/env python3
"""
Start ML Bot with Self-Evolution GUI on port 8501
Separate from main trading system
"""

import os
import sys
import subprocess
import time
import logging

def setup_logging():
    """Setup logging for ML Bot startup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn',
        'openai', 'trafilatura', 'feedparser', 'textblob'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def start_ml_bot_gui():
    """Start ML Bot GUI on port 8501"""
    logger = setup_logging()
    
    logger.info("🤖 Starting Self-Evolving ML Bot...")
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        logger.error(f"Missing dependencies: {missing}")
        logger.info("Please install missing packages using: pip install " + " ".join(missing))
        return False
    
    # Check OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("⚠️ OPENAI_API_KEY not found. Evolution features will be limited.")
    else:
        logger.info("✅ OpenAI API key detected. Full evolution features available.")
    
    # Change to ml_bot directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Start Streamlit GUI on port 8501
        logger.info("🚀 Launching ML Bot GUI on port 8501...")
        
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "ml_bot_gui.py", 
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ]
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info("✅ ML Bot GUI started successfully!")
        logger.info("🌐 Access at: http://localhost:8501")
        logger.info("📊 Features: Self-Evolution, Performance Monitoring, News Analysis")
        
        # Wait for a moment to check if process is running
        time.sleep(2)
        
        if process.poll() is None:
            logger.info("🔄 ML Bot is running... (Press Ctrl+C to stop)")
            # Keep the process running
            process.wait()
        else:
            # Process failed
            stdout, stderr = process.communicate()
            logger.error(f"❌ ML Bot failed to start:")
            logger.error(f"STDOUT: {stdout}")
            logger.error(f"STDERR: {stderr}")
            return False
            
    except KeyboardInterrupt:
        logger.info("🛑 ML Bot stopped by user")
        if 'process' in locals():
            process.terminate()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error starting ML Bot: {e}")
        return False

if __name__ == "__main__":
    success = start_ml_bot_gui()
    sys.exit(0 if success else 1)