import os
import sys
import pytest
import pandas as pd
import sqlite3
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_DIR, MODEL_DIR, STOCK_DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "financial_data.db")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "investment_classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


# ==================== Task 1: Data Collection ====================
class TestDataCollection:

    def test_csv_files_exist(self):
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        for ticker in tickers:
            path = os.path.join(STOCK_DATA_DIR, f"{ticker}_stock_data.csv")
            assert os.path.exists(path), f"Missing: {ticker}_stock_data.csv"

    def test_csv_has_required_columns(self):
        path = os.path.join(STOCK_DATA_DIR, "AAPL_stock_data.csv")
        df = pd.read_csv(path)
        required = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_csv_has_enough_rows(self):
        path = os.path.join(STOCK_DATA_DIR, "AAPL_stock_data.csv")
        df = pd.read_csv(path)
        assert len(df) >= 200, f"Too few rows: {len(df)}"

    def test_no_negative_prices(self):
        path = os.path.join(STOCK_DATA_DIR, "AAPL_stock_data.csv")
        df = pd.read_csv(path)
        assert (df['Close'] > 0).all(), "Negative/zero prices found"


# ==================== Task 2: Preprocessing ====================
class TestPreprocessing:

    def test_parquet_exists(self):
        parquet_path = os.path.join(DATA_DIR, "processed_stocks.parquet")
        assert os.path.exists(parquet_path), "Parquet file missing"

    def test_parquet_has_features(self):
        parquet_path = os.path.join(DATA_DIR, "processed_stocks.parquet")
        df = pd.read_parquet(parquet_path)
        required = ['MA_7', 'MA_30', 'MA_90', 'RSI', 'Volatility', 'Daily_Return', 'Sharpe_Ratio']
        for col in required:
            assert col in df.columns, f"Missing feature: {col}"

    def test_parquet_row_count(self):
        parquet_path = os.path.join(DATA_DIR, "processed_stocks.parquet")
        df = pd.read_parquet(parquet_path)
        assert len(df) >= 1000, f"Too few rows: {len(df)}"


# ==================== Task 3: Database ====================
class TestDatabase:

    def test_db_exists(self):
        assert os.path.exists(DB_PATH), "Database file missing"

    def test_table_exists(self):
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_data'")
            assert cursor.fetchone() is not None, "stock_data table missing"

    def test_db_has_data(self):
        with sqlite3.connect(DB_PATH) as conn:
            count = conn.execute("SELECT COUNT(*) FROM stock_data").fetchone()[0]
        assert count >= 1000, f"Too few rows in DB: {count}"

    def test_all_tickers_present(self):
        with sqlite3.connect(DB_PATH) as conn:
            tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM stock_data").fetchall()]
        for t in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
            assert t in tickers, f"Missing ticker: {t}"


# ==================== Task 5: Classifier ====================
class TestClassifier:

    def test_model_files_exist(self):
        assert os.path.exists(CLASSIFIER_PATH), "Classifier model missing"
        assert os.path.exists(SCALER_PATH), "Scaler missing"

    def test_model_loads(self):
        with open(CLASSIFIER_PATH, 'rb') as f:
            clf = pickle.load(f)
        assert clf is not None

    def test_prediction_valid(self):
        import numpy as np
        with open(CLASSIFIER_PATH, 'rb') as f:
            clf = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        features = np.array([[50.0, 150.0, 148.0, 145.0, 2.5, 0.01, 0.5]])
        scaled = scaler.transform(features)
        prediction = clf.predict(scaled)[0]
        assert prediction in ['BUY', 'HOLD', 'SELL'], f"Invalid prediction: {prediction}"

    def test_prediction_has_confidence(self):
        import numpy as np
        with open(CLASSIFIER_PATH, 'rb') as f:
            clf = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        features = np.array([[50.0, 150.0, 148.0, 145.0, 2.5, 0.01, 0.5]])
        scaled = scaler.transform(features)
        proba = clf.predict_proba(scaled)[0]
        assert abs(sum(proba) - 1.0) < 0.001, "Probabilities don't sum to 1"


# ==================== Task 7: Dashboard ====================
class TestDashboard:

    def test_dashboard_file_exists(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           "dashboard", "dashboard_app.py")
        assert os.path.exists(path), "Dashboard file missing"

    def test_chatbot_file_exists(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "chatbot", "ai_prediction_chatbot.py")
        assert os.path.exists(path), "Chatbot file missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])