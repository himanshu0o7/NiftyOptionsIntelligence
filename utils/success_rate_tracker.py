"""
Success Rate Tracker for Live Trading Performance
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
from utils.logger import Logger

class SuccessRateTracker:
    """Track and analyze trading success rates in real-time"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.logger = Logger()
        self._init_tracking_tables()
    
    def _init_tracking_tables(self):
        """Initialize tracking tables in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create performance tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    symbol TEXT,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    entry_price REAL,
                    exit_price REAL,
                    quantity INTEGER,
                    pnl REAL,
                    pnl_percentage REAL,
                    trade_result TEXT,  -- WIN/LOSS
                    exit_reason TEXT,   -- TP/SL/TSL/MANUAL
                    confidence_score REAL,
                    strategy_type TEXT,
                    market_mode TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create daily summary table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_summary (
                    date TEXT PRIMARY KEY,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    total_pnl REAL,
                    avg_win REAL,
                    avg_loss REAL,
                    max_drawdown REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing tracking tables: {e}")
    
    def record_trade(self, trade_data: Dict):
        """Record completed trade for performance tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate trade metrics
            pnl = (trade_data['exit_price'] - trade_data['entry_price']) * trade_data['quantity']
            pnl_percentage = (trade_data['exit_price'] - trade_data['entry_price']) / trade_data['entry_price'] * 100
            trade_result = "WIN" if pnl > 0 else "LOSS"
            
            cursor.execute('''
                INSERT OR REPLACE INTO performance_tracking 
                (trade_id, symbol, entry_time, exit_time, entry_price, exit_price, 
                 quantity, pnl, pnl_percentage, trade_result, exit_reason, 
                 confidence_score, strategy_type, market_mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('trade_id'),
                trade_data.get('symbol'),
                trade_data.get('entry_time'),
                trade_data.get('exit_time'),
                trade_data.get('entry_price'),
                trade_data.get('exit_price'),
                trade_data.get('quantity'),
                pnl,
                pnl_percentage,
                trade_result,
                trade_data.get('exit_reason', 'MANUAL'),
                trade_data.get('confidence_score', 0),
                trade_data.get('strategy_type', 'OPTIONS_BUY'),
                trade_data.get('market_mode', 'UNKNOWN')
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Trade recorded: {trade_result} | P&L: ₹{pnl:,.0f} | {trade_data['symbol']}")
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def get_current_stats(self, days: int = 30) -> Dict:
        """Get current performance statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get trades from last N days
            query = '''
                SELECT * FROM performance_tracking 
                WHERE entry_time >= datetime('now', '-{} days')
                ORDER BY entry_time DESC
            '''.format(days)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return self._empty_stats()
            
            # Calculate statistics
            total_trades = len(df)
            winning_trades = len(df[df['trade_result'] == 'WIN'])
            losing_trades = len(df[df['trade_result'] == 'LOSS'])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_pnl = df['pnl'].sum()
            avg_win = df[df['trade_result'] == 'WIN']['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = df[df['trade_result'] == 'LOSS']['pnl'].mean() if losing_trades > 0 else 0
            
            # Calculate drawdown
            df['cumulative_pnl'] = df['pnl'].cumsum()
            running_max = df['cumulative_pnl'].expanding().max()
            drawdown = df['cumulative_pnl'] - running_max
            max_drawdown = drawdown.min()
            
            # SL/TSL/TP trigger stats
            sl_triggers = len(df[df['exit_reason'].str.contains('SL', na=False)])
            tsl_triggers = len(df[df['exit_reason'].str.contains('TSL', na=False)])
            tp_triggers = len(df[df['exit_reason'].str.contains('TP', na=False)])
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 1),
                'total_pnl': round(total_pnl, 0),
                'avg_win': round(avg_win, 0),
                'avg_loss': round(avg_loss, 0),
                'max_drawdown': round(max_drawdown, 0),
                'sl_hit_percent': round((sl_triggers / total_trades * 100) if total_trades > 0 else 0, 1),
                'tsl_hit_percent': round((tsl_triggers / total_trades * 100) if total_trades > 0 else 0, 1),
                'tp_hit_percent': round((tp_triggers / total_trades * 100) if total_trades > 0 else 0, 1),
                'risk_reward_ratio': round(abs(avg_win / avg_loss) if avg_loss != 0 else 0, 2),
                'profit_factor': round(abs(df[df['trade_result'] == 'WIN']['pnl'].sum() / 
                                          df[df['trade_result'] == 'LOSS']['pnl'].sum()) if avg_loss != 0 else 0, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating stats: {e}")
            return self._empty_stats()
    
    def _empty_stats(self) -> Dict:
        """Return empty statistics structure"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'sl_hit_percent': 0,
            'tsl_hit_percent': 0,
            'tp_hit_percent': 0,
            'risk_reward_ratio': 0,
            'profit_factor': 0
        }
    
    def update_daily_summary(self):
        """Update daily summary table"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            stats = self.get_current_stats(days=1)  # Today's stats
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO daily_summary 
                (date, total_trades, winning_trades, losing_trades, win_rate, 
                 total_pnl, avg_win, avg_loss, max_drawdown)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                today,
                stats['total_trades'],
                stats['winning_trades'],
                stats['losing_trades'],
                stats['win_rate'],
                stats['total_pnl'],
                stats['avg_win'],
                stats['avg_loss'],
                stats['max_drawdown']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating daily summary: {e}")
    
    def get_strategy_performance(self) -> Dict:
        """Get performance by strategy type"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT strategy_type, market_mode, COUNT(*) as trades,
                       SUM(CASE WHEN trade_result = 'WIN' THEN 1 ELSE 0 END) as wins,
                       AVG(pnl) as avg_pnl,
                       SUM(pnl) as total_pnl
                FROM performance_tracking 
                WHERE entry_time >= datetime('now', '-30 days')
                GROUP BY strategy_type, market_mode
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            strategy_stats = {}
            for _, row in df.iterrows():
                key = f"{row['strategy_type']}_{row['market_mode']}"
                win_rate = (row['wins'] / row['trades'] * 100) if row['trades'] > 0 else 0
                
                strategy_stats[key] = {
                    'trades': row['trades'],
                    'win_rate': round(win_rate, 1),
                    'avg_pnl': round(row['avg_pnl'], 0),
                    'total_pnl': round(row['total_pnl'], 0)
                }
            
            return strategy_stats
            
        except Exception as e:
            self.logger.error(f"Error getting strategy performance: {e}")
            return {}
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades for display"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT symbol, entry_time, exit_time, pnl, pnl_percentage, 
                       trade_result, exit_reason, confidence_score
                FROM performance_tracking 
                ORDER BY exit_time DESC 
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(limit,))
            conn.close()
            
            return df.to_dict('records')
            
        except Exception as e:
            self.logger.error(f"Error getting recent trades: {e}")
            return []
    
    def export_performance_report(self, filepath: str = None):
        """Export detailed performance report"""
        try:
            if not filepath:
                filepath = f"performance_report_{datetime.now().strftime('%Y%m%d')}.csv"
            
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('SELECT * FROM performance_tracking ORDER BY entry_time DESC', conn)
            conn.close()
            
            df.to_csv(filepath, index=False)
            self.logger.info(f"Performance report exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting performance report: {e}")