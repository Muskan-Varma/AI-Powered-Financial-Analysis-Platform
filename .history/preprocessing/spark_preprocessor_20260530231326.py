"""
PySpark Data Preprocessing and Feature Engineering

STUDENT TASK (30 points):
Implement PySpark-based data preprocessing and feature engineering for stock data

LEARNING OBJECTIVES:
- Load data into Spark DataFrames
- Use Window functions for time series features
- Calculate technical indicators (MA, RSI, Volatility)
- Handle missing values
- Save processed data as Parquet

EXPECTED OUTPUT:
- Parquet file: data/processed_stocks.parquet
- Columns: Ticker, Date, Open, High, Low, Close, Volume,
           MA_7, MA_30, MA_90, RSI, Volatility, Daily_Return, Sharpe_Ratio
- ~6270 rows (1254 per ticker × 5 tickers)
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    STOCK_DATA_DIR, PROCESSED_DATA_DIR,
    MA_WINDOWS, RSI_WINDOW, VOLATILITY_WINDOW
)


class SparkPreprocessor:
    """PySpark-based data preprocessing and feature engineering"""

    def __init__(self, spark=None):
        """
        Initialize preprocessor with Spark session

        Args:
            spark: SparkSession (creates new one if None)

        TODO: Students - Complete initialization
        """
        if spark is None:
        raise NotImplementedError("TODO: Implement load_csv_files()")

    def calculate_moving_averages(self, df, windows=[7, 30, 90]):
        """
        Calculate moving averages using PySpark Window functions

        Args:
            df: Spark DataFrame
            windows: List of window sizes (e.g., [7, 30, 90])

        Returns:
            DataFrame with MA columns added

        TODO: Students - Implement this function (8 points)

        HINTS:
        1. Window specification: Window.partitionBy('Ticker').orderBy('Date').rowsBetween(-N+1, 0)
        2. For 7-day MA: rowsBetween(-6, 0) means current row + previous 6 rows = 7 total
        3. Use F.avg('Close').over(window) to calculate average
        4. Create columns: MA_7, MA_30, MA_90
        5. Loop through window sizes to avoid code duplication

        FORMULA:
            MA_N = Average of last N closing prices
        """
        print(f"\nCalculating moving averages: {windows}")

        # YOUR CODE HERE
        raise NotImplementedError("TODO: Implement calculate_moving_averages()")

    def calculate_rsi(self, df, window=14):
        """
        Calculate RSI (Relative Strength Index)

        Args:
            df: Spark DataFrame
            window: RSI window size (default: 14)

        Returns:
            DataFrame with RSI column added

        TODO: Students - Implement this function (10 points)

        RSI FORMULA (5 steps):
        1. Calculate price changes: Change = Close - Previous Close
        2. Separate gains and losses:
           - Gain = Change if Change > 0 else 0
           - Loss = -Change if Change < 0 else 0
        3. Calculate average gain and loss over window (14 days)
           - Avg Gain = Average of Gains over window
           - Avg Loss = Average of Losses over window
        4. Calculate RS = Avg Gain / Avg Loss
        5. Calculate RSI = 100 - (100 / (1 + RS))

        HINTS:
        1. Use F.lag('Close', 1) to get previous day's close
        2. Use F.when() for conditional logic (separate gains/losses)
        3. Use Window.partitionBy('Ticker').orderBy('Date') for each stock
        4. Use rolling window: rowsBetween(-(window-1), 0) for averages
        5. Drop intermediate columns at the end to keep DataFrame clean

        RSI INTERPRETATION:
        - RSI > 70: Overbought (potential sell signal)
        - RSI < 30: Oversold (potential buy signal)
        - RSI = 50: Neutral
        """
        print(f"\nCalculating RSI (window={window})...")

        # YOUR CODE HERE
        raise NotImplementedError("TODO: Implement calculate_rsi()")

    def calculate_volatility(self, df, window=30):
        """
        Calculate rolling volatility (standard deviation of close prices)

        Args:
            df: Spark DataFrame
            window: Volatility window size (default: 30)

        Returns:
            DataFrame with Volatility column added

        TODO: Students - Implement this function (3 points)

        FORMULA:
            Volatility = Standard Deviation of Close price over rolling window

        HINTS:
        1. Use Window.partitionBy('Ticker').orderBy('Date').rowsBetween(-(window-1), 0)
        2. Use F.stddev('Close').over(window) to calculate rolling std dev
        3. Higher volatility = more risky stock
        """
        print(f"\nCalculating volatility (window={window})...")

        # YOUR CODE HERE
        raise NotImplementedError("TODO: Implement calculate_volatility()")

    def calculate_returns(self, df):
        """
        Calculate daily returns and Sharpe ratio

        Args:
            df: Spark DataFrame

        Returns:
            DataFrame with Daily_Return and Sharpe_Ratio columns

        TODO: Students - Implement this function (4 points)

        DAILY RETURN FORMULA:
            Daily_Return = (Close - Previous Close) / Previous Close

        SHARPE RATIO FORMULA:
            Sharpe_Ratio = Mean(Daily_Return) / Std_Dev(Daily_Return)

        HINTS:
        1. Use lag() to get previous close
        2. Use Window functions for mean and stddev
        """
        print(f"\nCalculating returns and Sharpe ratio...")


        print(f"  ✅ Created Daily_Return and Sharpe_Ratio columns")
        return df

    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset

        Args:
            df: Spark DataFrame

        Returns:
            DataFrame with missing values handled

        TODO: Students - Implement missing value handling

        STRATEGIES:
        1. Drop rows with missing critical values (Open, High, Low, Close, Volume)
        2. Forward fill missing indicator values (MA, RSI, etc.)
        3. Or drop rows with any missing values

        HINTS:
        - Use df.dropna() to drop rows with null values
        - Or use df.na.fill() to fill with specific values
        """
        print(f"\nHandling missing values...")



        return df

    def save_to_parquet(self, df, output_path):
        """
        Save processed data to Parquet format

        Args:
            df: Processed Spark DataFrame
            output_path: Path to save Parquet file

        TODO: Students - Save DataFrame as Parquet (2 points)

        HINTS:
        - Use df.write.parquet()
        - Use mode='overwrite' to replace existing files
        - Parquet is columnar format, faster than CSV
        """
        print(f"\nSaving to Parquet: {output_path}")


        print(f"✅ Saved processed data to {output_path}")


def run_preprocessing(spark=None):
    """
    Main function to run preprocessing pipeline

    TODO: Students - Complete this function

    STEPS:
    1. Create SparkPreprocessor instance
    2. Load CSV files
    3. Run preprocessing
    4. Save to Parquet
    5. Show sample data
    """
    print("="*60)
    print("PYSPARK DATA PREPROCESSING")
    print("="*60)


    return processed_df


if __name__ == "__main__":
    """
    Run this file to test your preprocessing

    Usage: python preprocessing/spark_preprocessor.py
    """
    spark = SparkSession.builder \
        .appName("StockPreprocessing") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    run_preprocessing(spark)

    spark.stop()
