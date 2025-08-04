import logging

from telegram_alerts import send_telegram_alert

# Executes orders using SmartAPI

logger = logging.getLogger(__name__)


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
        logger.info("order_executor: Order placed: %s", response)
        return response
    except Exception as e:
        logger.exception("order_executor: Order failed: %s", e)
        send_telegram_alert(f"OrderExecutor error: {e}")
        raise

