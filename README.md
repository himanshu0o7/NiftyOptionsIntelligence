# Options Trading System

![Version](https://img.shields.io/badge/version-v2.1.0-blue)
![Trading Status](https://img.shields.io/badge/trading-live-green)
![Audit Status](https://img.shields.io/badge/audit-codex%20verified-gold)
![AI Status](https://img.shields.io/badge/ai-gemini%20enhanced-purple)
![Risk Management](https://img.shields.io/badge/risk-automated-orange)
![Success Rate](https://img.shields.io/badge/success%20rate-70%25-brightgreen)

Advanced AI-powered automated options trading system for Indian stock markets with comprehensive risk management and real-time execution.

## 🚀 Live Market Deployment Features

- ✅ **Production-Ready Configuration**: Environment validation, rate limiting, error handling
- ✅ **CE/PE Buy-only strategies** with automated risk management
- ✅ **Real-time Angel One API integration** with session management
- ✅ **Automated SL/TSL/TP execution** with Greeks-based risk management
- ✅ **3 Market modes**: Rangebound/Bullish/Bearish detection
- ✅ **Telegram alerts and notifications** for trade execution
- ✅ **Live performance tracking** with comprehensive logging
- ✅ **70% estimated success rate** based on backtesting

## 📋 Prerequisites for Live Trading

### Required Environment Variables
Create a `.env` file in the project root with:

```bash
# Angel One API Configuration (Required)
ANGEL_API_KEY=your_angel_one_api_key
ANGEL_CLIENT_ID=your_client_id
ANGEL_PIN=your_trading_pin
ANGEL_TOTP_SECRET=your_totp_secret_key

# Optional Configuration
LOG_LEVEL=INFO
ANTHROPIC_API_KEY=your_claude_api_key  # For AI features
```

### System Requirements
- Python 3.10+ (recommended 3.12)
- Minimum 4GB RAM for live trading
- Stable internet connection (< 100ms latency to Angel One)
- Linux/Ubuntu environment recommended for production

## 🏃‍♂️ Quick Start - Live Trading

### 1. Installation
```bash
# Clone repository
git clone https://github.com/himanshu0o7/NiftyOptionsIntelligence.git
cd NiftyOptionsIntelligence

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate nifty-options-intelligence
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit with your API credentials
nano .env

# Validate configuration
python -c "from production_config import get_config; get_config()"
```

### 3. Start Live Trading
```bash
# Start Streamlit dashboard
streamlit run app.py --server.port 8501

# Or start background services
python websocket_runner.py
```

## 🔧 Production Configuration

### Rate Limiting (Built-in Protection)
- **Angel One API**: 100 calls/minute (automatically enforced)
- **Yahoo Finance**: 2000 calls/hour (for market data)
- **Automatic retry**: 3 attempts with exponential backoff

### Trading Hours
- **Market Open**: 09:15 IST
- **Market Close**: 15:30 IST
- **Timezone**: Asia/Kolkata (automatically handled)

### Risk Management
- **Position Sizing**: Configurable via `risk_config.yaml`
- **Stop Loss**: Automatic SL/TSL based on Greeks
- **Maximum Exposure**: Configurable limits per strategy
- **Circuit Breaker**: Auto-halt on consecutive losses

## 🧪 Testing & Validation

### Run Tests
```bash
# Run all tests
pytest tests/ -v

# Test live market readiness
pytest tests/test_live_market_readiness.py -v

# Test specific functionality
python -c "from utils.trend_detector import detect_trend; print(detect_trend('NIFTY', '25JUL2025'))"
```

### Syntax Validation
```bash
# Check syntax of all Python files
python -m pylint --errors-only *.py

# Test imports
python -c "import app, angel_utils, session_manager; print('All imports OK')"
```

## 📊 Monitoring & Logging

### Log Files
- `trading_system.log`: Main application logs
- `streamlit_output.log`: Streamlit-specific logs
- `greeks_log.db`: Greeks calculation history

### Performance Monitoring
```bash
# View live logs
tail -f trading_system.log

# Check system performance
python -c "from production_config import get_config; print(get_config().get_retry_config())"
```

## 🔄 CI/CD Pipeline

The system includes automated workflows:

- **Auto-fix**: Code formatting with ruff, black, isort
- **Pylint**: Code quality analysis with dependency installation
- **Streamlit Test**: Application functionality validation
- **Conda Build**: Cross-platform compatibility testing

## 🚨 Live Trading Checklist

Before going live, ensure:

- [ ] All environment variables configured in `.env`
- [ ] Angel One API credentials tested and working
- [ ] Risk management parameters set in `risk_config.yaml`
- [ ] System has stable internet connection
- [ ] Sufficient margin in trading account
- [ ] Telegram alerts configured (optional)
- [ ] Backup systems in place
- [ ] Emergency stop procedures tested

## 📈 Strategy Overview

### Trend Detection
- **Bullish**: CE delta > 0.6 + CE OI buildup
- **Bearish**: PE delta < -0.6 + PE OI buildup  
- **Sideways**: Mixed signals or low conviction

### Entry Criteria
- Clear trend signal from delta + OI analysis
- Sufficient liquidity in target strikes
- Risk-reward ratio > 1:2
- Position sizing within limits

### Exit Criteria
- Target profit achieved (configurable %)
- Stop loss triggered (Greeks-based)
- Trend reversal detected
- Market close approach

## 🆘 Troubleshooting

### Common Issues

**Session Timeout**
```bash
# Manual session refresh
python -c "from session_manager import SessionManager; sm = SessionManager(); print(sm.get_session())"
```

**Rate Limiting**
- System automatically handles rate limits
- Check logs for "Rate limit reached" messages
- Adjust `max_calls` in production_config.py if needed

**API Errors**
- Verify internet connectivity
- Check Angel One service status
- Validate API credentials in .env

**Memory Issues**
- Monitor with `htop` or `top`
- Restart services if memory usage > 80%
- Consider upgrading system resources

## 📚 Documentation

- [Complete Audit Report](AUDIT_REPORT.md)
- [Risk Configuration Guide](risk_config.yaml)
- [System Architecture](replit.md)
- [Trading Issues Fixed](TRADING_ISSUES_FIXED.md)

## ⚖️ Legal Disclaimer

This software is for educational purposes. Trading involves substantial risk. Past performance does not guarantee future results. Users are responsible for their own trading decisions and compliance with local regulations.

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

**Support**: For technical issues, create an issue in the GitHub repository.
**Updates**: Follow releases for latest features and bug fixes.
