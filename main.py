from src.fetch_data import fetch_stock_data
from src.features import create_features
from src.train_model import train_xgboost
from src.predict import predict_next_day

def explain(pred, prob):
    if pred == 1:
        return f"📈 The model thinks the stock will GO UP tomorrow with {prob*100:.1f}% confidence."
    else:
        return f"📉 The model thinks the stock will GO DOWN tomorrow with {prob*100:.1f}% confidence."

if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g. AAPL, TCS.NS): ")

    print("\nFetching data...")
    raw_data = fetch_stock_data(ticker)

    print("Creating features...")
    data = create_features(raw_data)

    print("Training model...")
    train_xgboost(data)

    print("\nMaking prediction...")
    pred, prob = predict_next_day(data)

    print("\n--- Prediction Result ---")
    print(explain(pred, prob))
