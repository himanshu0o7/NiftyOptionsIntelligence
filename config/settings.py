import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class TradingSettings:
    """Trading configuration settings"""
    
    # API Settings
    API_BASE_URL: str = "https://apiconnect.angelone.in"
    WEBSOCKET_URL: str = "wss://smartapisocket.angelone.in/smart-stream"
    
    # Trading Parameters
    DEFAULT_QUANTITY: int = 1
    MAX_POSITIONS: int = 10
    DEFAULT_STOP_LOSS: float = 2.0  # Percentage
    DEFAULT_TAKE_PROFIT: float = 5.0  # Percentage
    
    # Risk Management
    MAX_DAILY_LOSS: float = 10000.0  # In Rupees
    MAX_POSITION_SIZE: float = 50000.0  # In Rupees
    RISK_PER_TRADE: float = 1.0  # Percentage of capital
    
    # Technical Indicators
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: int = 30
    RSI_OVERBOUGHT: int = 70
    
    EMA_SHORT: int = 9
    EMA_LONG: int = 21
    
    VWAP_PERIOD: int = 20
    
    # Options Trading
    NIFTY_LOT_SIZE: int = 50
    BANKNIFTY_LOT_SIZE: int = 15
    
    OPTION_EXPIRY_DAYS: List[str] = None
    
    # Database Settings
    DATABASE_PATH: str = "trading_data.db"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "trading_system.log"
    
    def __post_init__(self):
        if self.OPTION_EXPIRY_DAYS is None:
            self.OPTION_EXPIRY_DAYS = ["Thursday"]  # Weekly expiry
    
    @classmethod
    def from_env(cls) -> 'TradingSettings':
        """Create settings from environment variables"""
        return cls(
            DEFAULT_QUANTITY=int(os.getenv('DEFAULT_QUANTITY', '1')),
            MAX_POSITIONS=int(os.getenv('MAX_POSITIONS', '10')),
            DEFAULT_STOP_LOSS=float(os.getenv('DEFAULT_STOP_LOSS', '2.0')),
            DEFAULT_TAKE_PROFIT=float(os.getenv('DEFAULT_TAKE_PROFIT', '5.0')),
            MAX_DAILY_LOSS=float(os.getenv('MAX_DAILY_LOSS', '10000.0')),
            MAX_POSITION_SIZE=float(os.getenv('MAX_POSITION_SIZE', '50000.0')),
            RSI_PERIOD=int(os.getenv('RSI_PERIOD', '14')),
            EMA_SHORT=int(os.getenv('EMA_SHORT', '9')),
            EMA_LONG=int(os.getenv('EMA_LONG', '21')),
            VWAP_PERIOD=int(os.getenv('VWAP_PERIOD', '20')),
            LOG_LEVEL=os.getenv('LOG_LEVEL', 'INFO'),
        )

class Settings:
    """Main settings class"""
    
    def __init__(self):
        self.trading = TradingSettings.from_env()
        
        # Angel One API credentials from environment
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
        
    def validate(self) -> bool:
        """Validate critical settings"""
        if not self.paper_trading:
            if not all([self.api_key, self.client_code, self.password]):
                return False
        return True
    
    def get_strategy_config(self, strategy_name: str) -> Dict:
        """Get configuration for specific strategy"""
        base_config = {
            'stop_loss': self.trading.DEFAULT_STOP_LOSS,
            'take_profit': self.trading.DEFAULT_TAKE_PROFIT,
            'quantity': self.trading.DEFAULT_QUANTITY,
            'rsi_period': self.trading.RSI_PERIOD,
            'ema_short': self.trading.EMA_SHORT,
            'ema_long': self.trading.EMA_LONG,
        }
        
        # Strategy-specific configurations
        strategy_configs = {
            'breakout': {
                'breakout_threshold': 2.0,  # Percentage
                'volume_multiplier': 1.5,
                'confirmation_candles': 2,
            },
            'oi_analysis': {
                'oi_change_threshold': 20.0,  # Percentage
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
        
        if strategy_name in strategy_configs:
            base_config.update(strategy_configs[strategy_name])
        
        return base_config
