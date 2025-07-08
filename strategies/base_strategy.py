from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
from utils.logger import Logger
from core.database import Database

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.logger = Logger()
        self.db = Database()
        
        # Strategy state
        self.is_active = False
        self.positions = {}
        self.signals = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        # Risk management
        self.max_positions = config.get('max_positions', 5)
        self.position_size = config.get('position_size', 1)
        self.stop_loss = config.get('stop_loss', 2.0)  # Percentage
        self.take_profit = config.get('take_profit', 5.0)  # Percentage
        
    @abstractmethod
    def analyze(self, market_data: Dict) -> List[Dict]:
        """
        Analyze market data and generate signals
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def should_enter(self, symbol: str, data: Dict) -> Optional[Dict]:
        """
        Determine if strategy should enter a position
        
        Args:
            symbol: Trading symbol
            data: Market data for the symbol
            
        Returns:
            Entry signal if conditions are met, None otherwise
        """
        pass
    
    @abstractmethod
    def should_exit(self, symbol: str, position: Dict, current_data: Dict) -> Optional[Dict]:
        """
        Determine if strategy should exit a position
        
        Args:
            symbol: Trading symbol
            position: Current position data
            current_data: Current market data
            
        Returns:
            Exit signal if conditions are met, None otherwise
        """
        pass
    
    def add_position(self, symbol: str, position_data: Dict):
        """Add a new position"""
        self.positions[symbol] = position_data
        self.positions[symbol]['entry_time'] = datetime.now()
        self.positions[symbol]['strategy'] = self.name
        
        self.logger.info(f"[{self.name}] Added position: {symbol}")
    
    def remove_position(self, symbol: str) -> Optional[Dict]:
        """Remove a position"""
        if symbol in self.positions:
            position = self.positions.pop(symbol)
            position['exit_time'] = datetime.now()
            
            # Calculate P&L
            pnl = self._calculate_pnl(position)
            position['pnl'] = pnl
            
            # Update performance metrics
            self._update_performance(pnl)
            
            self.logger.info(f"[{self.name}] Removed position: {symbol}, P&L: {pnl}")
            return position
        
        return None
    
    def update_positions(self, market_data: Dict):
        """Update all positions with current market data"""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol].get('ltp', 0)
                position['current_price'] = current_price
                
                # Calculate unrealized P&L
                position['unrealized_pnl'] = self._calculate_unrealized_pnl(position, current_price)
                
                # Check exit conditions
                exit_signal = self.should_exit(symbol, position, market_data[symbol])
                if exit_signal:
                    self.signals.append(exit_signal)
    
    def _calculate_pnl(self, position: Dict) -> float:
        """Calculate realized P&L for a position"""
        entry_price = position.get('entry_price', 0)
        exit_price = position.get('exit_price', 0)
        quantity = position.get('quantity', 0)
        transaction_type = position.get('transaction_type', 'BUY')
        
        if transaction_type == 'BUY':
            pnl = (exit_price - entry_price) * quantity
        else:  # SELL
            pnl = (entry_price - exit_price) * quantity
        
        return round(pnl, 2)
    
    def _calculate_unrealized_pnl(self, position: Dict, current_price: float) -> float:
        """Calculate unrealized P&L for a position"""
        entry_price = position.get('entry_price', 0)
        quantity = position.get('quantity', 0)
        transaction_type = position.get('transaction_type', 'BUY')
        
        if transaction_type == 'BUY':
            pnl = (current_price - entry_price) * quantity
        else:  # SELL
            pnl = (entry_price - current_price) * quantity
        
        return round(pnl, 2)
    
    def _update_performance(self, pnl: float):
        """Update strategy performance metrics"""
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
    
    def check_risk_limits(self) -> bool:
        """Check if strategy is within risk limits"""
        # Check maximum positions
        if len(self.positions) >= self.max_positions:
            self.logger.warning(f"[{self.name}] Maximum positions limit reached")
            return False
        
        # Check daily loss limit
        daily_pnl = sum(pos.get('unrealized_pnl', 0) for pos in self.positions.values())
        max_daily_loss = self.config.get('max_daily_loss', 10000)
        
        if daily_pnl < -max_daily_loss:
            self.logger.warning(f"[{self.name}] Daily loss limit exceeded: {daily_pnl}")
            return False
        
        return True
    
    def apply_stop_loss(self, position: Dict, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        entry_price = position.get('entry_price', 0)
        transaction_type = position.get('transaction_type', 'BUY')
        stop_loss_pct = position.get('stop_loss', self.stop_loss)
        
        if transaction_type == 'BUY':
            stop_price = entry_price * (1 - stop_loss_pct / 100)
            return current_price <= stop_price
        else:  # SELL
            stop_price = entry_price * (1 + stop_loss_pct / 100)
            return current_price >= stop_price
    
    def apply_take_profit(self, position: Dict, current_price: float) -> bool:
        """Check if take profit should be triggered"""
        entry_price = position.get('entry_price', 0)
        transaction_type = position.get('transaction_type', 'BUY')
        take_profit_pct = position.get('take_profit', self.take_profit)
        
        if transaction_type == 'BUY':
            target_price = entry_price * (1 + take_profit_pct / 100)
            return current_price >= target_price
        else:  # SELL
            target_price = entry_price * (1 - take_profit_pct / 100)
            return current_price <= target_price
    
    def apply_trailing_stop(self, position: Dict, current_price: float) -> bool:
        """Apply trailing stop loss logic"""
        if not position.get('trailing_stop_enabled', False):
            return False
        
        entry_price = position.get('entry_price', 0)
        highest_price = position.get('highest_price', entry_price)
        lowest_price = position.get('lowest_price', entry_price)
        transaction_type = position.get('transaction_type', 'BUY')
        trailing_pct = position.get('trailing_stop', 1.0)  # Percentage
        
        if transaction_type == 'BUY':
            # Update highest price
            if current_price > highest_price:
                position['highest_price'] = current_price
                highest_price = current_price
            
            # Calculate trailing stop price
            trailing_stop_price = highest_price * (1 - trailing_pct / 100)
            return current_price <= trailing_stop_price
        
        else:  # SELL
            # Update lowest price
            if current_price < lowest_price:
                position['lowest_price'] = current_price
                lowest_price = current_price
            
            # Calculate trailing stop price
            trailing_stop_price = lowest_price * (1 + trailing_pct / 100)
            return current_price >= trailing_stop_price
    
    def get_performance_metrics(self) -> Dict:
        """Get strategy performance metrics"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_win = self.total_pnl / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = abs(self.total_pnl) / self.losing_trades if self.losing_trades > 0 else 0
        
        return {
            'strategy_name': self.name,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(self.total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'active_positions': len(self.positions)
        }
    
    def generate_signal(self, symbol: str, action: str, price: float, 
                       signal_type: str, confidence: float = 0.0, 
                       parameters: Dict = None) -> Dict:
        """Generate a trading signal"""
        signal = {
            'signal_id': f"{self.name}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'strategy_name': self.name,
            'symbol': symbol,
            'token': '',  # To be filled by calling code
            'signal_type': signal_type,
            'action': action,
            'price': price,
            'confidence': confidence,
            'parameters': parameters or {},
            'generated_at': datetime.now()
        }
        
        return signal
    
    def start(self):
        """Start the strategy"""
        self.is_active = True
        self.logger.info(f"[{self.name}] Strategy started")
    
    def stop(self):
        """Stop the strategy"""
        self.is_active = False
        self.logger.info(f"[{self.name}] Strategy stopped")
    
    def reset(self):
        """Reset strategy state"""
        self.positions.clear()
        self.signals.clear()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        self.logger.info(f"[{self.name}] Strategy reset")
