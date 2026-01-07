import joblib
import pandas as pd

def predict_next_day(latest_df):
    model = joblib.load("model/xgb_model.pkl")

    X_latest = latest_df[["return", "ma5", "ma10", "volatility"]].tail(1)

    prob = model.predict_proba(X_latest)[0][1]
    pred = model.predict(X_latest)[0]

    return pred, prob
