import numpy as np
import pandas as pd
from src.forecast import make_features


def test_features_do_not_contain_missing_rows():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    prices = pd.DataFrame({"Close": np.linspace(100, 130, len(dates))}, index=dates)
    result = make_features(prices)
    assert not result.isna().any().any()
    assert "target_next_return" in result.columns
