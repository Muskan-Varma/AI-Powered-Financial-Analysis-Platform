"""
Stock Data Downloader using Yahoo Finance API

STUDENT TASK (20 points):
Implement functions to download historical stock data from Yahoo Finance

WHAT YOU'LL LEARN:
- API integration with yfinance
- Error handling and data validation
- File I/O operations
- Data quality checks

EXPECTED OUTPUT:
- CSV files in data/stock_data/
- Format: {TICKER}_stock_data.csv
- Columns: Ticker, Date, Open, High, Low, Close, Volume
- ~1254 rows per ticker (5 years daily data)
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import STOCK_DATA_DIR, DEFAULT_TICKERS, DEFAULT_START_DATE


class StockDownloader:
    """Downloads historical stock data from Yahoo Finance"""

    def __init__(self, tickers=None, start_date=None, end_date=None):
        """
        Initialize downloader

        TODO: Set tickers, dates, create output directory
        """
        self.tickers = tickers or DEFAULT_TICKERS
        self.start_date = start_date or DEFAULT_START_DATE
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        os.makedirs(STOCK_DATA_DIR, exist_ok=True)

    def download_stock_data(self, ticker):
        """
        Download data for ONE ticker

        Args:
            ticker: Stock symbol (e.g., 'AAPL')

        Returns:
            DataFrame with columns: Ticker, Date, Open, High, Low, Close, Volume

        TODO (10 points):
        1. Use yf.download(ticker, start=..., end=...)
        2. Check if data is empty
        3. Reset index to make Date a column
        4. Add 'Ticker' column
        5. Return DataFrame

        HINT: yf.download returns DataFrame with Date as index
        """
        print(f"Downloading {ticker}...")

        # YOUR CODE HERE
        raise NotImplementedError("TODO: Implement download_stock_data()")

    def download_all(self):
        """
        Download data for ALL tickers

        Returns:
            dict: {ticker: DataFrame}

        TODO (5 points):
        1. Loop through self.tickers
        2. Call download_stock_data() for each
        3. Store in dictionary
        4. Handle failures gracefully
        """
        stock_data = {}

        # YOUR CODE HERE
        raise NotImplementedError("TODO: Implement download_all()")

    def save_to_csv(self, stock_data):
        """
        Save all data to CSV files

        Args:
            stock_data: dict of {ticker: DataFrame}

        TODO (3 points):
        1. Loop through stock_data
        2. Create filename: f"{ticker}_stock_data.csv"
        3. Save: df.to_csv(filepath, index=False)
        """
        # YOUR CODE HERE
        raise NotImplementedError("TODO: Implement save_to_csv()")

    def validate_data(self, stock_data):
        """
        Validate data quality

        Args:
            stock_data: dict of DataFrames

        Returns:
            bool: True if all valid

        TODO (2 points):
        Check for:
        - Minimum 200 rows
        - No missing values in critical columns
        - No duplicate dates
        - Positive prices
        - No extreme outliers (>50% daily change)
        """
        # YOUR CODE HERE
        return True  # Change after implementing


def run_data_collection():
    """
    Main function - orchestrates download process

    TODO:
    1. Create StockDownloader()
    2. Call download_all()
    3. Call validate_data()
    4. Call save_to_csv()
    5. Print summary
    """
    print("="*60)
    print("STOCK DATA COLLECTION")
    print("="*60)

    # YOUR CODE HERE
    raise NotImplementedError("TODO: Complete run_data_collection()")


if __name__ == "__main__":
    run_data_collection()
