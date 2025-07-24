# Executes orders using SmartAPI
def place_order(order_data, smart_api_obj):
    """
    order_data = {
        "variety": "NORMAL",
        "tradingsymbol": "NIFTY24JUL23500CE",
        "symboltoken": "123456",
        "transactiontype": "BUY",
        "exchange": "NSE",
        "ordertype": "LIMIT",
        "producttype": "INTRADAY",
        "duration": "DAY",
        "price": 10,
        "quantity": 75
    }
    """
    try:
        response = smart_api_obj.placeOrder(order_data)
        print("✅ Order placed:", response)
        return response
    except Exception as e:
        print("❌ Order failed:", e)
        return None

