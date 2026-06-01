"""
Spark MLlib Gradient Boosted Trees for Time Series Forecasting

STUDENT TASK (35 points):
Implement time series forecasting using Spark MLlib GBT Regressor

LEARNING OBJECTIVES:
- Create lagged features for time series
- Train Gradient Boosted Trees model
- Make multi-step predictions
- Evaluate model performance

EXPECTED OUTPUT:
- Trained model saved to data/models/spark_gbt_forecaster/
- Test R² Score: 0.90+ (90%+ accuracy)
- Test RMSE: $25-40
- Mean % Error: < 10%
"""

from pyspark.sql import SparkSession
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, lag
from pyspark.sql.window import Window
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MODEL_DIR, LOOKBACK_DAYS, FORECAST_DAYS, \
    GBT_MAX_ITER, GBT_MAX_DEPTH, GBT_LEARNING_RATE


class SparkGBTForecaster:
    """
    Time Series Forecasting using Spark MLlib Gradient Boosted Trees

    TODO: Students - Complete all methods marked with TODO
    """

    def __init__(self, spark=None, lookback_days=30, forecast_days=7):
        """
        Initialize forecaster

        Args:
            spark: SparkSession
            lookback_days: Number of historical days to use as features
            forecast_days: Number of days to forecast

        TODO: Complete initialization
        """
        self.spark = spark
        self.lookback_days = lookback_days
        self.forecast_days = forecast_days
        self.model = None
        self.pipeline_model = None
        self.feature_columns = []

    def create_lagged_features(self, df):
        """
        Create lagged features for time series forecasting

        Args:
            df: Spark DataFrame with columns [Ticker, Date, Open, High, Low, Close, Volume]

        Returns:
            DataFrame with lagged features

        TODO: Students - Implement this function (15 points)

        TASK:
        Create 150 lagged features:
        - Close_lag_1 to Close_lag_30 (30 features)
        - Open_lag_1 to Open_lag_30 (30 features)
        - High_lag_1 to High_lag_30 (30 features)
        - Low_lag_1 to Low_lag_30 (30 features)
        - Volume_lag_1 to Volume_lag_30 (30 features)

        IMPORTANT: Create features in BATCHES to avoid StackOverflowError!

        HINTS:
        1. Use Window.partitionBy('Ticker').orderBy('Date')
        2. Use lag(col, N) to get value N days ago
        3. Create in batches: 1-10, 11-20, 21-30
        4. Use .cache() and .count() after each batch

        EXAMPLE:
            window = Window.partitionBy('Ticker').orderBy('Date')
            for lag_days in range(1, 11):  # Batch 1: lag 1-10
                df = df.withColumn(f'Close_lag_{lag_days}',
                                   lag(col('Close'), lag_days).over(window))
            df = df.cache()
            df.count()  # Force computation

            # Repeat for batches 11-20 and 21-30
        """
        print(f"Creating {self.lookback_days}-day lagged features...")


        print(f"Created {len(self.feature_columns)} lagged features")
        return df

    def train(self, spark_df):
        """
        Train the GBT model

        Args:
            spark_df: Spark DataFrame with features

        Returns:
            dict: Training metrics

        TODO: Students - Implement training (10 points)

        STEPS:
        1. Create lagged features
        2. Rename 'target' to 'label'
        3. Split data: 80% train, 10% val, 10% test
        4. Create VectorAssembler
        5. Create GBTRegressor
        6. Create Pipeline
        7. Train model
        8. Evaluate on train/val/test sets

        HINTS:
        - Use VectorAssembler to combine features into single vector
        - Use GBTRegressor with hyperparameters from config
        - Use RegressionEvaluator for metrics (RMSE, R², MAE)
        """
        print("Training Gradient Boosted Trees model...")


        return metrics

    def predict_future(self, df, ticker, num_days=7):
        """
        Predict future stock prices

        Args:
            df: Spark DataFrame with historical data
            ticker: Stock ticker
            num_days: Number of days to predict

        Returns:
            Pandas DataFrame with predictions

        TODO: Students - Implement prediction (10 points)

        STEPS:
        1. Filter data for specific ticker
        2. Create lagged features
        3. Get most recent features
        4. Make prediction
        5. Add realistic variation between days

        IMPORTANT: Add 1% random variation to avoid same predictions
        """
        pass

    def save_model(self, filepath):
        """Save trained model"""
        self.pipeline_model.write().overwrite().save(filepath)
        print(f"Model saved: {filepath}")

    def load_model(self, filepath):
        """Load trained model"""
        from pyspark.ml import PipelineModel
        self.pipeline_model = PipelineModel.load(filepath)
        print(f"Model loaded: {filepath}")


if __name__ == "__main__":
    """
    Test your forecaster

    Usage: python ml_models/spark_gbt_forecaster.py
    """
    print("TODO: Implement and test Spark GBT Forecaster")
