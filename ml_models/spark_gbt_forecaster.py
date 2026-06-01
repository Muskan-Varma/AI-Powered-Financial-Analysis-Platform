import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import PROCESSED_DATA_DIR, MODEL_DIR, LOOKBACK_DAYS, FORECAST_DAYS, GBT_MAX_ITER, GBT_MAX_DEPTH, GBT_LEARNING_RATE


class SparkGBTForecaster:

    def __init__(self, spark=None):
        if spark is None:
            self.spark = SparkSession.builder \
                .appName("StockForecasting") \
                .config("spark.driver.memory", "4g") \
                .getOrCreate()
        else:
            self.spark = spark
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.models = {}

    def load_data(self, parquet_path):
        print("Loading processed data...")
        df = self.spark.read.parquet(parquet_path)
        print(f"  Loaded {df.count()} rows")
        return df

    def create_features(self, df, ticker):
        print(f"  Creating features for {ticker}...")
        df = df.filter(F.col('Ticker') == ticker).orderBy('Date')
        w = Window.partitionBy('Ticker').orderBy('Date')

        for lag in range(1, LOOKBACK_DAYS + 1):
            df = df.withColumn(f'lag_{lag}', F.lag('Close', lag).over(w))

        for lag in range(1, 8):
            df = df.withColumn(f'rsi_lag_{lag}', F.lag('RSI', lag).over(w))
            df = df.withColumn(f'vol_lag_{lag}', F.lag('Volatility', lag).over(w))

        df = df.withColumn('target', F.lead('Close', FORECAST_DAYS).over(w))
        df = df.dropna()
        return df

    def prepare_train_test(self, df):
        feature_cols = [c for c in df.columns if c.startswith('lag_') or
                       c.startswith('rsi_lag_') or c.startswith('vol_lag_') or
                       c in ['MA_7', 'MA_30', 'MA_90', 'RSI', 'Volatility', 'Sharpe_Ratio']]

        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
        df = assembler.transform(df)

        total = df.count()
        train_size = int(total * 0.8)
        df_sorted = df.orderBy('Date')
        train = df_sorted.limit(train_size)
        test = df_sorted.subtract(train)
        return train, test, feature_cols

    def train_model(self, train_df):
        print("  Training GBT model...")
        gbt = GBTRegressor(
            featuresCol='features',
            labelCol='target',
            maxIter=GBT_MAX_ITER,
            maxDepth=GBT_MAX_DEPTH,
            stepSize=GBT_LEARNING_RATE
        )
        model = gbt.fit(train_df)
        return model

    def evaluate_model(self, model, test_df, ticker):
        predictions = model.transform(test_df)
        evaluator = RegressionEvaluator(labelCol='target', predictionCol='prediction')
        rmse = evaluator.evaluate(predictions, {evaluator.metricName: 'rmse'})
        mae = evaluator.evaluate(predictions, {evaluator.metricName: 'mae'})
        r2 = evaluator.evaluate(predictions, {evaluator.metricName: 'r2'})
        print(f"  {ticker} — RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

    def save_model(self, model, ticker):
        path = os.path.join(MODEL_DIR, f"gbt_{ticker}")
        model.write().overwrite().save(path)
        print(f"  Model saved: {path}")

    def train_all(self, parquet_path):
        df = self.load_data(parquet_path)
        tickers = [row['Ticker'] for row in df.select('Ticker').distinct().collect()]
        results = {}
        for ticker in tickers:
            print(f"\n{'='*40}")
            print(f"Training: {ticker}")
            ticker_df = self.create_features(df, ticker)
            train, test, _ = self.prepare_train_test(ticker_df)
            model = self.train_model(train)
            metrics = self.evaluate_model(model, test, ticker)
            self.save_model(model, ticker)
            self.models[ticker] = model
            results[ticker] = metrics
        return results


def run_forecasting(spark=None):
    print("="*60)
    print("GBT STOCK FORECASTING")
    print("="*60)
    forecaster = SparkGBTForecaster(spark)
    results = forecaster.train_all(PROCESSED_DATA_DIR)
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for ticker, metrics in results.items():
        print(f"{ticker}: RMSE={metrics['rmse']:.4f} | R2={metrics['r2']:.4f}")
    return forecaster


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("StockForecasting") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    run_forecasting(spark)
    spark.stop()