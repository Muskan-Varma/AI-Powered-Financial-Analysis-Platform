import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import STOCK_DATA_DIR, PROCESSED_DATA_DIR, MA_WINDOWS, RSI_WINDOW, VOLATILITY_WINDOW


class SparkPreprocessor:

    def __init__(self, spark=None):
        if spark is None:
            self.spark = SparkSession.builder \
                .appName("StockPreprocessing") \
                .config("spark.driver.memory", "4g") \
                .getOrCreate()
        else:
            self.spark = spark

    def load_csv_files(self, data_dir):
        print("\nLoading CSV files...")
        dfs = []
        for file in os.listdir(data_dir):
            if file.endswith("_stock_data.csv"):
                path = os.path.join(data_dir, file)
                df = self.spark.read.csv(path, header=True, inferSchema=True)
                dfs.append(df)
                print(f"  Loaded: {file}")
        combined = dfs[0]
        for df in dfs[1:]:
            combined = combined.union(df)
        print(f"  Total rows: {combined.count()}")
        return combined

    def calculate_moving_averages(self, df, windows=[7, 30, 90]):
        print(f"\nCalculating moving averages: {windows}")
        for w in windows:
            window_spec = Window.partitionBy('Ticker').orderBy('Date').rowsBetween(-(w-1), 0)
            df = df.withColumn(f'MA_{w}', F.avg('Close').over(window_spec))
        return df

    def calculate_rsi(self, df, window=14):
        print(f"\nCalculating RSI (window={window})...")
        w = Window.partitionBy('Ticker').orderBy('Date')
        df = df.withColumn('prev_close', F.lag('Close', 1).over(w))
        df = df.withColumn('change', F.col('Close') - F.col('prev_close'))
        df = df.withColumn('gain', F.when(F.col('change') > 0, F.col('change')).otherwise(0))
        df = df.withColumn('loss', F.when(F.col('change') < 0, -F.col('change')).otherwise(0))
        roll = Window.partitionBy('Ticker').orderBy('Date').rowsBetween(-(window-1), 0)
        df = df.withColumn('avg_gain', F.avg('gain').over(roll))
        df = df.withColumn('avg_loss', F.avg('loss').over(roll))
        df = df.withColumn('rs', F.when(F.col('avg_loss') == 0, 100).otherwise(F.col('avg_gain') / F.col('avg_loss')))
        df = df.withColumn('RSI', 100 - (100 / (1 + F.col('rs'))))
        df = df.drop('prev_close', 'change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs')
        return df

    def calculate_volatility(self, df, window=30):
        print(f"\nCalculating volatility (window={window})...")
        roll = Window.partitionBy('Ticker').orderBy('Date').rowsBetween(-(window-1), 0)
        df = df.withColumn('Volatility', F.stddev('Close').over(roll))
        return df

    def calculate_returns(self, df):
        print(f"\nCalculating returns and Sharpe ratio...")
        w = Window.partitionBy('Ticker').orderBy('Date')
        df = df.withColumn('prev_close', F.lag('Close', 1).over(w))
        df = df.withColumn('Daily_Return', (F.col('Close') - F.col('prev_close')) / F.col('prev_close'))
        roll = Window.partitionBy('Ticker').orderBy('Date').rowsBetween(-29, 0)
        df = df.withColumn('mean_return', F.avg('Daily_Return').over(roll))
        df = df.withColumn('std_return', F.stddev('Daily_Return').over(roll))
        df = df.withColumn('Sharpe_Ratio', F.when(F.col('std_return') == 0, 0).otherwise(F.col('mean_return') / F.col('std_return')))
        df = df.drop('prev_close', 'mean_return', 'std_return')
        return df

    def handle_missing_values(self, df):
        print(f"\nHandling missing values...")
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        df = df.dropna()
        return df

    def save_to_parquet(self, df, output_path):
        print(f"\nSaving to Parquet: {output_path}")
        df.write.mode('overwrite') \
            .option("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false") \
            .parquet(output_path)
        print(f"Saved processed data to {output_path}")


def run_preprocessing(spark=None):
    print("="*60)
    print("PYSPARK DATA PREPROCESSING")
    print("="*60)
    preprocessor = SparkPreprocessor(spark)
    df = preprocessor.load_csv_files(STOCK_DATA_DIR)
    df = preprocessor.calculate_moving_averages(df, MA_WINDOWS)
    df = preprocessor.calculate_rsi(df, RSI_WINDOW)
    df = preprocessor.calculate_volatility(df, VOLATILITY_WINDOW)
    df = preprocessor.calculate_returns(df)
    df = preprocessor.handle_missing_values(df)
    preprocessor.save_to_parquet(df, PROCESSED_DATA_DIR)
    print(f"\nFinal rows: {df.count()}")
    print(f"Columns: {df.columns}")
    df.show(5)
    return df


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("StockPreprocessing") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    run_preprocessing(spark)
    spark.stop()