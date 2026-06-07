import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
STOCK_DATA_DIR = os.path.join(DATA_DIR, "stock_data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_stocks.parquet")
MODEL_DIR = os.path.join(DATA_DIR, "models")

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
DEFAULT_START_DATE = "2020-01-01"

MA_WINDOWS = [7, 30, 90]
RSI_WINDOW = 14
VOLATILITY_WINDOW = 30

LOOKBACK_DAYS = 30
FORECAST_DAYS = 7

GBT_MAX_ITER = 150
GBT_MAX_DEPTH = 4
GBT_LEARNING_RATE = 0.05