# 📈 Stock Trend Prediction using XGBoost

This project predicts whether a stock will go **UP or DOWN** the next trading day using historical price data and an XGBoost machine learning model.

## 🔹 Features
- Fetches real-time stock data using Yahoo Finance
- Creates technical indicators:
  - Daily returns
  - Moving averages
  - Volatility
- Trains an XGBoost classifier
- Predicts next-day direction with confidence score
- Designed to be understandable for non-technical users

## 🔹 Tech Stack
- Python
- XGBoost
- Pandas, NumPy
- Scikit-learn
- yFinance

## 🔹 How to Run
```bash
pip install -r requirements.txt
python main.py
