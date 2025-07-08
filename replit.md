# Options Trading System

## Overview

This is a comprehensive options trading system built with Python and Streamlit that provides real-time market data analysis, automated trading strategies, risk management, and portfolio monitoring capabilities. The system integrates with Angel One's SmartAPI for live market data and trade execution, supports both paper trading and live trading modes, and includes advanced technical analysis and Greeks calculations for options trading.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application with multi-page dashboard
- **UI Components**: Interactive charts using Plotly, real-time data displays, configuration forms
- **Pages Structure**:
  - Main trading dashboard with live charts and options chain
  - P&L analysis and performance tracking
  - Risk monitoring and alerts
  - Strategy configuration and backtesting
- **State Management**: Streamlit session state for maintaining user data and connections

### Backend Architecture
- **Core API Integration**: Angel One SmartAPI client for market data and order execution
- **WebSocket Client**: Real-time market data streaming via Angel One WebSocket
- **Database Layer**: SQLite database for storing market data, orders, positions, and historical records
- **Strategy Engine**: Modular strategy framework with base strategy class and specific implementations
- **Risk Management**: Position tracking, risk calculation, and automated risk controls

### Data Storage Solutions
- **Primary Database**: SQLite for local data persistence
- **Tables**: market_data, orders, positions, signals, risk_metrics
- **Caching**: JSON files for instrument master data and symbol mappings
- **Real-time Data**: In-memory storage with periodic database persistence

## Key Components

### Trading Core (`core/`)
- **AngelAPI**: Authentication, order management, historical data retrieval
- **WebSocketClient**: Live market data streaming with automatic reconnection
- **Database**: SQLite operations with schema management and data persistence

### Trading Strategies (`strategies/`)
- **BaseStrategy**: Abstract base class defining strategy interface
- **BreakoutStrategy**: Identifies price breakouts with volume confirmation
- **OIAnalysis**: Open Interest analysis for options trading signals
- **Strategy Framework**: Pluggable architecture for adding new strategies

### Risk Management (`risk_management/`)
- **PositionManager**: Track positions, P&L, and position-level risk
- **RiskCalculator**: VaR calculations, portfolio risk metrics, stress testing
- **Risk Controls**: Automated stop-loss, position sizing, exposure limits

### Technical Analysis (`indicators/`)
- **TechnicalIndicators**: RSI, EMA, VWAP, and other technical indicators
- **GreeksCalculator**: Black-Scholes options pricing and Greeks calculation
- **Market Analysis**: Support/resistance identification, trend analysis

### Data Management (`data/`, `symbols/`)
- **InstrumentManager**: Download and cache instrument master data
- **Symbol Mappings**: Pre-configured NIFTY and BANKNIFTY option symbols
- **Market Data**: Historical and real-time data processing

## Data Flow

1. **Market Data Ingestion**: WebSocket client receives live ticks → Database storage → Strategy analysis
2. **Signal Generation**: Strategies analyze market data → Generate buy/sell signals → Risk validation
3. **Order Execution**: Validated signals → Angel One API → Order placement → Position tracking
4. **Risk Monitoring**: Continuous position monitoring → Risk calculations → Alert generation
5. **P&L Tracking**: Real-time P&L updates → Performance metrics → Dashboard display

## External Dependencies

### API Integrations
- **Angel One SmartAPI**: Primary broker integration for Indian markets
- **Real-time Data**: WebSocket connection for live market feeds
- **Authentication**: TOTP-based two-factor authentication

### Python Libraries
- **Streamlit**: Web application framework
- **Plotly**: Interactive charting and visualization
- **Pandas/NumPy**: Data manipulation and numerical computing
- **TA-Lib**: Technical analysis indicators
- **SciPy**: Statistical calculations for risk metrics
- **WebSocket-client**: Real-time data streaming
- **PyOTP**: TOTP authentication support

### Data Sources
- **Instrument Master**: Angel One's OpenAPI instrument file
- **Market Data**: Live and historical price data via API
- **Options Chain**: Real-time options data including Greeks and OI

## Deployment Strategy

### Local Development
- **Environment**: Python 3.8+ with virtual environment
- **Database**: SQLite for local data storage
- **Configuration**: Environment variables for API credentials
- **Logging**: File-based logging with configurable levels

### Production Considerations
- **Scalability**: Modular architecture supports horizontal scaling
- **Reliability**: Automatic reconnection for WebSocket connections
- **Security**: API credentials stored securely, no hardcoded secrets
- **Monitoring**: Comprehensive logging and error handling
- **Backup**: Database backup strategies for trade history

### Trading Modes
- **Paper Trading**: Default mode for testing strategies without real money
- **Live Trading**: Real order execution with proper risk controls
- **Simulation**: Historical backtesting capabilities

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

- **July 08, 2025**: Live trading configuration complete
  - Configured system for ₹17,000 capital with 1 lot trading
  - Implemented comprehensive risk management (5% daily loss limit, 20% position size)
  - Angel One API integration with manual TOTP authentication
  - Fixed TOTP secret encoding issues and missing utilities
  - System now ready for live trading with proper risk controls

## Trading Configuration

### Live Trading Setup
- **Capital**: ₹17,000 total trading capital
- **Risk Management**: 
  - Daily loss limit: ₹850 (5% of capital)
  - Position size limit: ₹3,400 (20% per position)
  - Risk per trade: ₹340 (2% of capital)
  - Maximum concurrent positions: 5
  - Lot size: 1 lot only
- **Trading Mode**: Live trading enabled (paper trading disabled)
- **API Integration**: Angel One SmartAPI with secure credential storage

## Changelog

- July 08, 2025: Initial setup and live trading configuration