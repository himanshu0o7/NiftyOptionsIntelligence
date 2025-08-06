def get_option_greek_data(symbol, expiry, option_type):
    return {"delta": 0.72 if option_type == "CE" else -0.65}

