from smart_websocket_handler import SmartWebSocketHandler
from angel_utils import load_master_contract

sws = SmartWebSocketHandler()

nfo_df = load_master_contract()
nifty_options = nfo_df[(nfo_df['name'] == 'NIFTY') & (nfo_df['instrumenttype'] == 'OPTIDX')]
latest_expiry = sorted(nifty_options['expiry'].unique())[0]

atm_strike = int(round(nifty_options['strike'].mean() / 100) * 100)
ce_token = str(nifty_options[(nifty_options['strike'] == atm_strike * 100) & (nifty_options['expiry'] == latest_expiry) & (nifty_options['symbol'].str.endswith('CE'))]['token'].iloc[0])
pe_token = str(nifty_options[(nifty_options['strike'] == atm_strike * 100) & (nifty_options['expiry'] == latest_expiry) & (nifty_options['symbol'].str.endswith('PE'))]['token'].iloc[0])

live_tokens = [ce_token, pe_token]
token_list = [{"exchangeType": 2, "tokens": live_tokens}]
sws.start_websocket(token_list=token_list, mode=2)

if __name__ == "__main__":
    import time
    try:
        while True:
            for token in live_tokens:
                data = sws.get_latest_data(token)
                if data:
                    print(f"Token: {token} | LTP: {data['ltp']} | OI: {data['oi']} | Volume: {data['volume']}")
            time.sleep(5)
    except KeyboardInterrupt:
        sws.stop_websocket()