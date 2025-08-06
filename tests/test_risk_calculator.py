import pytest
import pandas as pd
import numpy as np
from risk_management.risk_calculator import RiskCalculator

@pytest.fixture
def dummy_config():
    return {}

@pytest.fixture
def calculator(dummy_config) -> RiskCalculator:
    return RiskCalculator(config=dummy_config)

def test_calculate_risk_returns_dataframe(calculator):
    df = pd.DataFrame([
        {"Symbol": "NIFTY", "Strike": 17200, "Qty": 50, "LTP": 135.0}
    ])
    result = calculator.calculate_risk(df)
    assert "Delta" in result.columns
    assert not result.empty
