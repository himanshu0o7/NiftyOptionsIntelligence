"""
config/settings.py

Loads environment variables and defines configuration settings
for trading, alerts, and system behavior.
"""

import os
from dataclasses import dataclass
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class TradingSettings:
    """Trading configuration settings"""

    # API
    API_BASE_URL: str = "https://apiconnect.angelone.in"
    WEBSOCKET_URL: str = "wss://smartapisocket.angelone.in/smart-stream"

    # Capital & Position Sizing
    CAPITAL: float = 17000.0
    DEFAULT_QUANTITY: int = 1
    MAX_POSITIONS: int = 5
    MAX_DAILY_LOSS: float = 850.0
    MAX_POSITION_SIZE: float = 3400.0
    RISK_PER_TRADE: float = 2.0

    # Risk Controls
    DEFAULT_STOP_LOSS: float = 2.0
    DEFAULT_TAKE_PROFIT: float = 5.0

    # Indicators
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: int = 30
    RSI_OVERBOUGHT: int = 70
    EMA_SHORT: int = 9
    EMA_LONG: int = 21
    VWAP_PERIOD: int = 20

    # Options
    NIFTY_LOT_SIZE: int = 75
    BANKNIFTY_LOT_SIZE: int = 15
    FINNIFTY_LOT_SIZE: int = 25
    MIDCPNIFTY_LOT_SIZE: int = 50
    NIFTYNXT50_LOT_SIZE: int = 120
    SUPPORTED_INDICES: Dict[str, int] = None
    OPTION_EXPIRY_DAYS: List[str] = None

    # DB and Logging
    DATABASE_PATH: str = "trading_data.db"
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "trading_system.log"

    def __post_init__(self):
        if self.SUPPORTED_INDICES is None:
            self.SUPPORTED_INDICES = {
                'NIFTY': 75,
                'BANKNIFTY': 15,
                'FINNIFTY': 25,
                'MIDCPNIFTY': 50,
                'NIFTYNXT50': 120
            }
        if self.OPTION_EXPIRY_DAYS is None:
            self.OPTION_EXPIRY_DAYS = ["Thursday"]

    @classmethod
    def from_env(cls) -> 'TradingSettings':
        """Initialize from environment variables (optional override)."""
        return cls(
            DEFAULT_QUANTITY=int(os.getenv('DEFAULT_QUANTITY', 1)),
            MAX_POSITIONS=int(os.getenv('MAX_POSITIONS', 5)),
            DEFAULT_STOP_LOSS=float(os.getenv('DEFAULT_STOP_LOSS', 2.0)),
            DEFAULT_TAKE_PROFIT=float(os.getenv('DEFAULT_TAKE_PROFIT', 5.0)),
            MAX_DAILY_LOSS=float(os.getenv('MAX_DAILY_LOSS', 850.0)),
            MAX_POSITION_SIZE=float(os.getenv('MAX_POSITION_SIZE', 3400.0)),
            RSI_PERIOD=int(os.getenv('RSI_PERIOD', 14)),
            EMA_SHORT=int(os.getenv('EMA_SHORT', 9)),
            EMA_LONG=int(os.getenv('EMA_LONG', 21)),
            VWAP_PERIOD=int(os.getenv('VWAP_PERIOD', 20)),
            LOG_LEVEL=os.getenv('LOG_LEVEL', 'INFO'),
        )


class Settings:
    """Main application configuration class"""

    def __init__(self):
        # Trading settings
        self.trading = TradingSettings.from_env()

        # Angel One API credentials
        self.api_key = os.getenv('ANGEL_API_KEY', '')
        self.client_code = os.getenv('ANGEL_CLIENT_CODE', '')
        self.password = os.getenv('ANGEL_PASSWORD', '')

        # System settings
        self.paper_trading = os.getenv('PAPER_TRADING', 'True').lower() == 'true'
        self.enable_websocket = os.getenv('ENABLE_WEBSOCKET', 'True').lower() == 'true'
        self.auto_trade = os.getenv('AUTO_TRADE', 'False').lower() == 'true'

        # Market timings (IST)
        self.market_start_time = "09:15"
        self.market_end_time = "15:30"

        # Strategy settings
        self.enabled_strategies = os.getenv('ENABLED_STRATEGIES', 'breakout,oi_analysis').split(',')

        # Telegram alerting
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')

    def validate(self) -> bool:
        """Validate required credentials for live trading."""
        if not self.paper_trading and not all([self.api_key, self.client_code, self.password]):
            return False
        return True

    def get_strategy_config(self, strategy_name: str) -> Dict:
        """Return base + strategy-specific configuration."""
        base_config = {
            'stop_loss': self.trading.DEFAULT_STOP_LOSS,
            'take_profit': self.trading.DEFAULT_TAKE_PROFIT,
            'quantity': self.trading.DEFAULT_QUANTITY,
            'rsi_period': self.trading.RSI_PERIOD,
            'ema_short': self.trading.EMA_SHORT,
            'ema_long': self.trading.EMA_LONG,
        }

        strategy_overrides = {
            'breakout': {
                'breakout_threshold': 2.0,
                'volume_multiplier': 1.5,
                'confirmation_candles': 2,
            },
            'oi_analysis': {
                'oi_change_threshold': 20.0,
                'price_oi_divergence': True,
                'max_pain_analysis': True,
            },
            'greeks_based': {
                'delta_threshold': 0.5,
                'gamma_threshold': 0.05,
                'theta_threshold': -50,
                'vega_threshold': 20,
            }
        }

        if strategy_name in strategy_overrides:
            base_config.update(strategy_overrides[strategy_name])
        return base_config


# Enable importing Settings globally
__all__ = ["Settings", "TradingSettings"]

