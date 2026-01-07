import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start="2025-01-01", end="2026-01-01"):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data
