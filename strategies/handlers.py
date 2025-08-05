# strategies/handlers.py

def get_signal_from_data(data):
    # Dummy logic
    if data['price'] > data['moving_avg']:
        return "BUY_CE"
    elif data['price'] < data['moving_avg']:
        return "SELL_PE"
    else:
        return "HOLD"

