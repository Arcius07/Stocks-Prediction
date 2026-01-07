import pandas as pd

def create_features(df):
    df = df.copy()

    # Daily return
    df["return"] = df["Close"].pct_change()

    # Moving averages
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma10"] = df["Close"].rolling(10).mean()

    # Volatility
    df["volatility"] = df["return"].rolling(5).std()

    # Target: 1 if tomorrow goes UP else 0
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    return df
