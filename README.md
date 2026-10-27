# S&P 500 Forecasting Lab

A reproducible financial-machine-learning project comparing a historical
baseline with a tree-based model for next-day S&P 500 returns.

## Research question

Can lagged returns, rolling volatility, and moving-average features improve
next-day return predictions without leaking future information?

## Method

1. Download daily S&P 500 data with `yfinance`.
2. Create lagged return, rolling-volatility, and moving-average features.
3. Split chronologically rather than randomly.
4. Compare a naive zero-return baseline with Random Forest regression.
5. Report MAE, RMSE, and directional accuracy.

This is an educational research project, not investment advice. Financial
markets are non-stationary and out-of-sample performance is not guaranteed.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.forecast
pytest
```

No market data is committed; the script downloads it at runtime.
