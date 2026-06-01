import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import STOCK_DATA_DIR, DEFAULT_TICKERS, DEFAULT_START_DATE


class StockDownloader:

    def __init__(self, tickers=None, start_date=None, end_date=None):
        self.tickers = tickers or DEFAULT_TICKERS
        self.start_date = start_date or DEFAULT_START_DATE
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        os.makedirs(STOCK_DATA_DIR, exist_ok=True)

    def download_stock_data(self, ticker):
        print(f"Downloading {ticker}...")
        try:
            df = yf.download(ticker, start=self.start_date, end=self.end_date, auto_adjust=True)
            if df.empty:
                print(f"  No data for {ticker}")
                return None
            df.reset_index(inplace=True)
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df['Ticker'] = ticker
            df = df[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            print(f"  {ticker}: {len(df)} rows")
            return df
        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")
            return None

    def download_all(self):
        stock_data = {}
        for ticker in self.tickers:
            df = self.download_stock_data(ticker)
            if df is not None:
                stock_data[ticker] = df
        return stock_data

    def save_to_csv(self, stock_data):
        for ticker, df in stock_data.items():
            filepath = os.path.join(STOCK_DATA_DIR, f"{ticker}_stock_data.csv")
            df.to_csv(filepath, index=False)
            print(f"  Saved: {filepath}")

    def validate_data(self, stock_data):
        all_valid = True
        for ticker, df in stock_data.items():
            if len(df) < 200:
                print(f"  {ticker}: Too few rows ({len(df)})")
                all_valid = False
            if df[['Open','High','Low','Close','Volume']].isnull().any().any():
                print(f"  {ticker}: Missing values found")
                all_valid = False
            if df['Date'].duplicated().any():
                print(f"  {ticker}: Duplicate dates found")
                all_valid = False
            if (df['Close'] <= 0).any():
                print(f"  {ticker}: Negative/zero prices found")
                all_valid = False
        if all_valid:
            print("  All data valid!")
        return all_valid


def run_data_collection():
    print("="*60)
    print("STOCK DATA COLLECTION")
    print("="*60)
    downloader = StockDownloader()
    stock_data = downloader.download_all()
    downloader.validate_data(stock_data)
    downloader.save_to_csv(stock_data)
    print("\nData collection complete!")
    print(f"Files saved in: {STOCK_DATA_DIR}")


if __name__ == "__main__":
    run_data_collection()