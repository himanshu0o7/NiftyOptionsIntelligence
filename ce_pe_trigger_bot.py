import time
import logging
from smart_websocket_handler import SmartWebSocketHandler
from angel_utils import load_master_contract

logger = logging.getLogger(__name__)
sws = SmartWebSocketHandler()

class CEPETriggerBot:
    def __init__(self, symbol="NIFTY"):
        self.symbol = symbol
        self.token_map = {}
        self.active_tokens = []

    def initialize_tokens(self):
        df = load_master_contract()
        df = df[(df['name'] == self.symbol) & (df['instrumenttype'] == 'OPTIDX')]
        expiry = sorted(df['expiry'].unique())[0]
        atm_strike = int(round(df['strike'].mean() / 100) * 100)

        ce = df[(df['strike'] == atm_strike) & (df['expiry'] == expiry) & (df['symbol'].str.endswith('CE'))].iloc[0]
        pe = df[(df['strike'] == atm_strike) & (df['expiry'] == expiry) & (df['symbol'].str.endswith('PE'))].iloc[0]

        self.token_map = {
            str(ce['token']): {"symbol": ce['symbol'], "strike": atm_strike, "type": "CE"},
            str(pe['token']): {"symbol": pe['symbol'], "strike": atm_strike, "type": "PE"}
        }
        self.active_tokens = list(self.token_map.keys())

    def start_stream(self):
        self.initialize_tokens()
        token_list = [{"exchangeType": 2, "tokens": self.active_tokens}]
        sws.start_websocket(token_list=token_list, mode=2)

    def run_loop(self):
        logger.info("🟢 Starting CE/PE trigger loop")
        try:
            while True:
                for token in self.active_tokens:
                    tick = sws.get_latest_data(token)
                    meta = self.token_map[token]
                    if tick and tick.get('volume', 0) > 500000 and tick.get('oi', 0) > 100000:
                        print(f"🚨 Trigger Alert: {meta['symbol']} | Strike: {meta['strike']} | Type