# Nifty Options Intelligence Bot - Live Trading Deployment Guide

## Prerequisites
1. Angel One Demat Account with API access
2. Python 3.8+ installed
3. Stable internet connection
4. VPS/Server for 24/7 operation (recommended)

## Environment Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd NiftyOptionsIntelligence
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your Angel One credentials
```

### 4. Verify Setup
```bash
python test_bot_functionality.py
```

## Live Trading Setup

### 1. Start the Bot
```bash
./start_bot.sh
```

### 2. Access Dashboard
Open http://localhost:8501 in your browser

### 3. Monitor Performance
```bash
python monitor_bot.py
```

## Production Checklist

- [ ] Environment variables configured
- [ ] Risk management parameters set
- [ ] Logging directories created
- [ ] Database files initialized
- [ ] Monitoring alerts configured
- [ ] Error handling verified
- [ ] Performance optimizations applied

## Troubleshooting

### Common Issues
1. **Connection Errors**: Check internet connectivity and Angel One API status
2. **Authentication Failures**: Verify API credentials and TOTP setup
3. **WebSocket Disconnections**: The bot has auto-reconnection logic
4. **Database Errors**: Check file permissions and disk space

### Log Locations
- Trading logs: `logs/trading/`
- WebSocket logs: `logs/websocket/`
- Strategy logs: `logs/strategies/`

## Support
For issues and support, check the documentation and logs first.

## Disclaimer
This bot is for educational purposes. Use at your own risk. Always test with small amounts first.
