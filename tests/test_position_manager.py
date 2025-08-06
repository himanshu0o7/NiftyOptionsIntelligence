import pytest
import pandas as pd
from risk_management.position_manager import PositionManager

@pytest.fixture
def dummy_config():
    return {}

@pytest.fixture
def position_manager(dummy_config) -> PositionManager:
    return PositionManager(config=dummy_config)
