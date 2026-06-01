import sqlite3
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "financial_data.db")
PARQUET_PATH = os.path.join(DATA_DIR, "processed_stocks.parquet")


class DatabaseManager:

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        os.makedirs(DATA_DIR, exist_ok=True)

    def create_tables(self):
        print("Creating tables...")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    ticker TEXT NOT NULL,
                    date DATETIME NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume INTEGER,
                    ma_7 REAL, ma_30 REAL, ma_90 REAL,
                    rsi REAL, volatility REAL, daily_return REAL, sharpe_ratio REAL,
                    PRIMARY KEY (ticker, date)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON stock_data(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON stock_data(date)")
            conn.commit()
        print("  Tables created!")

    def load_from_parquet(self, parquet_path=None):
        path = parquet_path or PARQUET_PATH
        print(f"Loading parquet: {path}")
        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql('stock_data', conn, if_exists='replace', index=False)
        print(f"  Loaded {len(df)} rows into database!")
        return len(df)

    def get_stock_data(self, ticker, days=30):
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM stock_data
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(ticker, days))
        return df

    def get_latest_prices(self):
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT ticker, date, close, rsi, ma_7, ma_30, daily_return
                FROM stock_data
                WHERE date = (SELECT MAX(date) FROM stock_data WHERE ticker = s.ticker)
                FROM stock_data s
                GROUP BY ticker
                ORDER BY ticker
            """
            df = pd.read_sql_query(query, conn)
        return df

    def get_all_data(self, ticker):
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM stock_data WHERE ticker = ? ORDER BY date",
                conn, params=(ticker,)
            )
        return df


def run_database_setup():
    print("="*60)
    print("SQLITE DATABASE SETUP")
    print("="*60)
    db = DatabaseManager()
    db.create_tables()
    rows = db.load_from_parquet()
    print(f"\nVerifying data...")
    df = db.get_stock_data('AAPL', 5)
    print(df[['ticker','date','close','rsi','ma_7']].to_string(index=False))
    print(f"\nDatabase ready at: {DB_PATH}")
    return db


if __name__ == "__main__":
    run_database_setup()