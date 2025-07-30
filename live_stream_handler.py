import logging
import time
from datetime import datetime
import pytz
import pandas as pd
from smart_websocket_handler import SmartWebSocketHandler
from angel_utils import load_nfo_scrip_master

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LiveStreamHandler:
    def __init__(self):
        self.sws = SmartWebSocketHandler()
        self.token_lists = []

    def is_market_open(self):
        """Check if NSE market is open (9:15 AM - 3:30 PM IST, Mon-Fri)."""
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        if now.weekday() >= 5:  # Sat/Sun
            return False
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_open <= now <= market_close

    def fetch_tokens(self, symbol):
        """Fetch ATM CE/PE tokens for given symbol (NIFTY or BANKNIFTY)."""
        try:
            df = load_nfo_scrip_master()
            df = df[(df['name'] == symbol) & (df['instrumenttype'] == 'OPTIDX')]
            if df.empty:
                logger.error(f"No options data for {symbol} in scrip master.")
                return []
            
            logger.info(f"Raw {symbol} DF shape: {df.shape}")
            
            # Convert expiry to datetime for safe sorting
            df['expiry_dt'] = pd.to_datetime(df['expiry'], format='%d-%b-%Y', errors='coerce')
            valid_expiries = df['expiry_dt'].dropna().unique()
            if len(valid_expiries) == 0:
                logger.error(f"No valid expiry dates for {symbol}.")
                return []
            nearest_expiry = sorted(valid_expiries)[0].strftime('%d-%b-%Y')
            
            # Strike in paise (e.g., 2480000 for 24800)
            atm_strike = round(df['strike'].mean() / 10000) * 100
            atm_strike_paise = atm_strike * 100
            
            logger.info(f"Fetching for {symbol}, strike {atm_strike}, expiry {nearest_expiry}")
            
            ce_df = df[(df['strike'] == atm_strike_paise) & (df['expiry'] == nearest_expiry) & (df['symbol'].str.endswith('CE'))]
            pe_df = df[(df['strike'] == atm_strike_paise) & (df['expiry'] == nearest_expiry) & (df['symbol'].str.endswith('PE'))]
            
            if ce_df.empty or pe_df.empty:
                logger.error(f"No ATM CE/PE for {symbol} at strike {atm_strike}, expiry {nearest_expiry}. CE shape: {ce_df.shape}, PE shape: {pe_df.shape}")
                return []
            
            ce_token = str(ce_df['token'].iloc[0])
            pe_token = str(pe_df['token'].iloc[0])
            logger.info(f"Fetched tokens for {symbol}: CE={ce_token}, PE={pe_token}")
            return [{"exchangeType": 2, "tokens": [ce_token, pe_token]}]
        except Exception as e:
            logger.error(f"Error fetching tokens for {symbol}: {e}")
            return []

    def start(self):
        if not self.is_market_open():
            logger.warning("Market is closed. Using cached data or skipping live stream.")
        
        # Fetch for NIFTY and BANKNIFTY
        self.token_lists = self.fetch_tokens("NIFTY") + self.fetch_tokens("BANKNIFTY")
        if not self.token_lists:
            logger.error("No tokens fetched. Cannot start WebSocket.")
            return
        
        self.sws.start_websocket(self.token_lists, mode=2)
        logger.info("WebSocket started for NIFTY and BANKNIFTY ATM options.")

    def run(self):
        try:
            while True:
                if not self.is_market_open():
                    logger.warning("Market closed. Stopping stream.")
                    break
                all_data = self.sws.get_all_latest_data()
                if not all_data:
                    logger.warning("No live data received.")
                for token, data in all_data.items():
                    logger.info(f"Token: {token} | LTP: {data.get('ltp', 'N/A')} | OI: {data.get('oi', 'N/A')} | Volume: {data.get('volume', 'N/A')}")
                time.sleep(5)
        except KeyboardInterrupt:
            self.sws.stop_websocket()
            logger.info("Stream stopped.")

if __name__ == "__main__":
    stream = LiveStreamHandler()
    stream.start()
    stream.run()

