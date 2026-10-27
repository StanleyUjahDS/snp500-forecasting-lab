"""S&P 500 return forecasting with leakage-aware chronological evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def make_features(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["Close"].astype(float)
    returns = close.pct_change()
    frame = pd.DataFrame(index=prices.index)
    frame["return_lag_1"] = returns.shift(1)
    frame["return_lag_5"] = returns.shift(5)
    frame["volatility_20"] = returns.rolling(20).std().shift(1)
    frame["moving_average_gap_20"] = (close / close.rolling(20).mean() - 1).shift(1)
    frame["target_next_return"] = returns.shift(-1)
    return frame.dropna()


def evaluate(frame: pd.DataFrame) -> dict[str, float]:
    x = frame.drop(columns="target_next_return")
    y = frame["target_next_return"]
    split = int(len(frame) * 0.8)
    model = RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=5, n_jobs=-1)
    model.fit(x.iloc[:split], y.iloc[:split])
    prediction = model.predict(x.iloc[split:])
    actual = y.iloc[split:]
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "directional_accuracy": float((np.sign(actual) == np.sign(prediction)).mean()),
    }


def main() -> None:
    prices = yf.download("^GSPC", period="10y", auto_adjust=True, progress=False)
    if prices.empty:
        raise RuntimeError("No market data was returned by yfinance")
    print(evaluate(make_features(prices)))


if __name__ == "__main__":
    main()
