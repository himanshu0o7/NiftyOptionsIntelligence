# utils/historical.py

import pandas as pd
import datetime as dt

def get_historical_candles(symbol):
    """
    Mock historical data — replace with Angel One candle API.
    """
    now = dt.datetime.now()
    data = {
        "datetime": [now - dt.timedelta(minutes=i*5) for i in range(20)],
        "open": [17400 + i*2 for i in range(20)],
        "high": [17405 + i*2 for i in range(20)],
        "low": [17395 + i*2 for i in range(20)],
        "close": [17403 + i*2 for i in range(20)],
        "volume": [1000 + i*10 for i in range(20)]
    }
    df = pd.DataFrame(data)
    df = df.sort_values("datetime", ascending=True).reset_index(drop=True)
    return df
