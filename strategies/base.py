from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    def __init__(self, symbol, data):
        self.symbol = symbol
        self.data = data
        self.signal_handlers = {
            "BUY_CE": self._buy_call_option,
            "SELL_PE": self._sell_put_option,
        }

    @abstractmethod
    def should_enter(self):
        pass

    @abstractmethod
    def should_exit(self):
        pass

    def on_signal(self, signal: str):
        handler = self.signal_handlers.get(signal)
        if not handler:
            raise ValueError(f"Unknown signal: {signal}")
        return handler()

    def _buy_call_option(self):
        return f"Buying Call Option for {self.symbol}"

    def _sell_put_option(self):
        return f"Selling Put Option for {self.symbol}"

