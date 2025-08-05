"""Tests for the strategy_config thin wrapper."""

import importlib


def test_wrapper_exposes_page_function():
    wrapper = importlib.import_module("strategy_config")
    page = importlib.import_module("pages.strategy_config")
    assert wrapper.show_strategy_config is page.show_strategy_config
