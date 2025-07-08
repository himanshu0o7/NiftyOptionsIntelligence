import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import json
from utils.logger import Logger

class Database:
    """Database manager for trading system data"""
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.logger = Logger()
        self._init_database()
    
    def _init_database(self):
        """Initialize database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Market data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        token TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        open_price REAL,
                        high_price REAL,
                        low_price REAL,
                        close_price REAL,
                        volume INTEGER,
                        open_interest INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, token, timestamp)
                    )
                ''')
                
                # Orders table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        order_id TEXT UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        token TEXT NOT NULL,
                        transaction_type TEXT NOT NULL,
                        order_type TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        price REAL,
                        trigger_price REAL,
                        product_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        exchange TEXT NOT NULL,
                        order_tag TEXT,
                        strategy_name TEXT,
                        placed_at DATETIME NOT NULL,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        execution_details TEXT
                    )
                ''')
                
                # Positions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        token TEXT NOT NULL,
                        product_type TEXT NOT NULL,
                        exchange TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        avg_price REAL NOT NULL,
                        current_price REAL,
                        pnl REAL,
                        unrealized_pnl REAL,
                        realized_pnl REAL,
                        strategy_name TEXT,
                        opened_at DATETIME NOT NULL,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        closed_at DATETIME,
                        status TEXT DEFAULT 'OPEN'
                    )
                ''')
                
                # Trades table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT UNIQUE NOT NULL,
                        order_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        token TEXT NOT NULL,
                        transaction_type TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        price REAL NOT NULL,
                        exchange TEXT NOT NULL,
                        trade_time DATETIME NOT NULL,
                        strategy_name TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (order_id) REFERENCES orders (order_id)
                    )
                ''')
                
                # Signals table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_id TEXT UNIQUE NOT NULL,
                        symbol TEXT NOT NULL,
                        token TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        action TEXT NOT NULL,
                        price REAL NOT NULL,
                        confidence REAL,
                        strategy_name TEXT NOT NULL,
                        parameters TEXT,
                        generated_at DATETIME NOT NULL,
                        executed BOOLEAN DEFAULT FALSE,
                        executed_at DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Strategy performance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS strategy_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        date DATE NOT NULL,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        gross_pnl REAL DEFAULT 0,
                        net_pnl REAL DEFAULT 0,
                        max_drawdown REAL DEFAULT 0,
                        win_rate REAL DEFAULT 0,
                        avg_win REAL DEFAULT 0,
                        avg_loss REAL DEFAULT 0,
                        sharpe_ratio REAL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(strategy_name, date)
                    )
                ''')
                
                # Options data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS options_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        underlying TEXT NOT NULL,
                        strike_price REAL NOT NULL,
                        expiry_date DATE NOT NULL,
                        option_type TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        token TEXT NOT NULL,
                        ltp REAL,
                        open_interest INTEGER,
                        volume INTEGER,
                        iv REAL,
                        delta REAL,
                        gamma REAL,
                        theta REAL,
                        vega REAL,
                        rho REAL,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, token, updated_at)
                    )
                ''')
                
                # System logs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        log_level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        module TEXT,
                        function TEXT,
                        line_number INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_orders_symbol_status ON orders(symbol, status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol_trade_time ON trades(symbol, trade_time)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_strategy_generated_at ON signals(strategy_name, generated_at)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_options_data_underlying_expiry ON options_data(underlying, expiry_date)')
                
                conn.commit()
                # Only log once during first initialization
                # self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def insert_market_data(self, market_data: List[Dict]):
        """Insert market data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for data in market_data:
                    cursor.execute('''
                        INSERT OR REPLACE INTO market_data 
                        (symbol, token, timestamp, open_price, high_price, low_price, close_price, volume, open_interest)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data['symbol'],
                        data['token'],
                        data['timestamp'],
                        data.get('open', 0),
                        data.get('high', 0),
                        data.get('low', 0),
                        data.get('close', 0),
                        data.get('volume', 0),
                        data.get('open_interest', 0)
                    ))
                
                conn.commit()
                self.logger.debug(f"Inserted {len(market_data)} market data records")
                
        except Exception as e:
            self.logger.error(f"Error inserting market data: {str(e)}")
    
    def insert_order(self, order_data: Dict):
        """Insert order data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO orders 
                    (order_id, symbol, token, transaction_type, order_type, quantity, price, 
                     trigger_price, product_type, status, exchange, order_tag, strategy_name, 
                     placed_at, execution_details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    order_data['order_id'],
                    order_data['symbol'],
                    order_data['token'],
                    order_data['transaction_type'],
                    order_data['order_type'],
                    order_data['quantity'],
                    order_data.get('price'),
                    order_data.get('trigger_price'),
                    order_data['product_type'],
                    order_data['status'],
                    order_data['exchange'],
                    order_data.get('order_tag'),
                    order_data.get('strategy_name'),
                    order_data['placed_at'],
                    json.dumps(order_data.get('execution_details', {}))
                ))
                
                conn.commit()
                self.logger.debug(f"Inserted order: {order_data['order_id']}")
                
        except Exception as e:
            self.logger.error(f"Error inserting order: {str(e)}")
    
    def update_position(self, position_data: Dict):
        """Update position data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO positions 
                    (symbol, token, product_type, exchange, quantity, avg_price, current_price,
                     pnl, unrealized_pnl, realized_pnl, strategy_name, opened_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    position_data['symbol'],
                    position_data['token'],
                    position_data['product_type'],
                    position_data['exchange'],
                    position_data['quantity'],
                    position_data['avg_price'],
                    position_data.get('current_price'),
                    position_data.get('pnl', 0),
                    position_data.get('unrealized_pnl', 0),
                    position_data.get('realized_pnl', 0),
                    position_data.get('strategy_name'),
                    position_data['opened_at'],
                    position_data.get('status', 'OPEN')
                ))
                
                conn.commit()
                self.logger.debug(f"Updated position: {position_data['symbol']}")
                
        except Exception as e:
            self.logger.error(f"Error updating position: {str(e)}")
    
    def insert_signal(self, signal_data: Dict):
        """Insert trading signal"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO signals 
                    (signal_id, symbol, token, signal_type, action, price, confidence,
                     strategy_name, parameters, generated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data['signal_id'],
                    signal_data['symbol'],
                    signal_data['token'],
                    signal_data['signal_type'],
                    signal_data['action'],
                    signal_data['price'],
                    signal_data.get('confidence'),
                    signal_data['strategy_name'],
                    json.dumps(signal_data.get('parameters', {})),
                    signal_data['generated_at']
                ))
                
                conn.commit()
                self.logger.debug(f"Inserted signal: {signal_data['signal_id']}")
                
        except Exception as e:
            self.logger.error(f"Error inserting signal: {str(e)}")
    
    def get_market_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get market data for analysis"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM market_data 
                    WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                '''
                
                df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                return df
                
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return pd.DataFrame()
    
    def get_active_positions(self) -> pd.DataFrame:
        """Get active positions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM positions 
                    WHERE status = 'OPEN'
                    ORDER BY opened_at DESC
                '''
                
                return pd.read_sql_query(query, conn)
                
        except Exception as e:
            self.logger.error(f"Error fetching active positions: {str(e)}")
            return pd.DataFrame()
    
    def get_recent_signals(self, hours: int = 24) -> pd.DataFrame:
        """Get recent signals"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                
                query = '''
                    SELECT * FROM signals 
                    WHERE generated_at >= ?
                    ORDER BY generated_at DESC
                '''
                
                return pd.read_sql_query(query, conn, params=(cutoff_time,))
                
        except Exception as e:
            self.logger.error(f"Error fetching recent signals: {str(e)}")
            return pd.DataFrame()
    
    def get_strategy_performance(self, strategy_name: str, days: int = 30) -> pd.DataFrame:
        """Get strategy performance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = datetime.now().date() - timedelta(days=days)
                
                query = '''
                    SELECT * FROM strategy_performance 
                    WHERE strategy_name = ? AND date >= ?
                    ORDER BY date DESC
                '''
                
                return pd.read_sql_query(query, conn, params=(strategy_name, cutoff_date))
                
        except Exception as e:
            self.logger.error(f"Error fetching strategy performance: {str(e)}")
            return pd.DataFrame()
    
    def update_options_data(self, options_data: List[Dict]):
        """Update options data with Greeks"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for data in options_data:
                    cursor.execute('''
                        INSERT OR REPLACE INTO options_data 
                        (underlying, strike_price, expiry_date, option_type, symbol, token,
                         ltp, open_interest, volume, iv, delta, gamma, theta, vega, rho)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data['underlying'],
                        data['strike_price'],
                        data['expiry_date'],
                        data['option_type'],
                        data['symbol'],
                        data['token'],
                        data.get('ltp'),
                        data.get('open_interest'),
                        data.get('volume'),
                        data.get('iv'),
                        data.get('delta'),
                        data.get('gamma'),
                        data.get('theta'),
                        data.get('vega'),
                        data.get('rho')
                    ))
                
                conn.commit()
                self.logger.debug(f"Updated {len(options_data)} options data records")
                
        except Exception as e:
            self.logger.error(f"Error updating options data: {str(e)}")
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old data to maintain database size"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Clean old market data
                cursor.execute('DELETE FROM market_data WHERE created_at < ?', (cutoff_date,))
                
                # Clean old logs
                cursor.execute('DELETE FROM system_logs WHERE created_at < ?', (cutoff_date,))
                
                # Clean old signals (keep longer for analysis)
                signal_cutoff = datetime.now() - timedelta(days=days*2)
                cursor.execute('DELETE FROM signals WHERE created_at < ?', (signal_cutoff,))
                
                conn.commit()
                self.logger.info(f"Cleaned up data older than {days} days")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {str(e)}")
