"""
Init module for config package to allow absolute imports.
"""
from .settings import Settings, TradingSettings  # noqa: F401
__all__ = ["Settings", "TradingSettings"]

