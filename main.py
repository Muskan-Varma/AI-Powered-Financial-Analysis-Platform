"""
AI-Powered Financial Analysis Platform
Main Pipeline Orchestrator
"""

import os
import sys
from pyspark.sql import SparkSession

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_spark():
    return SparkSession.builder \
        .appName("FinancialAnalysisPlatform") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()


def main():
    print("=" * 60)
    print("AI-POWERED FINANCIAL ANALYSIS PLATFORM")
    print("=" * 60)

    # Task 1: Data Collection
    print("\n[1/5] DATA COLLECTION...")
    from data_collection.stock_downloader import run_data_collection
    run_data_collection()

    # Task 2: PySpark Preprocessing
    print("\n[2/5] PYSPARK PREPROCESSING...")
    spark = create_spark()
    from preprocessing.spark_preprocessor import run_preprocessing
    run_preprocessing(spark)

    # Task 3: SQLite Database
    print("\n[3/5] DATABASE SETUP...")
    from sql_interface.database_manager import run_database_setup
    run_database_setup()

    # Task 4: GBT Forecasting
    print("\n[4/5] GBT FORECASTING...")
    from ml_models.spark_gbt_forecaster import run_forecasting
    run_forecasting(spark)

    spark.stop()

    # Task 5: Investment Classifier
    print("\n[5/5] INVESTMENT CLASSIFIER...")
    from ml_models.investment_classifier import run_classification
    run_classification()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nTo start Dashboard:")
    print("  streamlit run dashboard/dashboard_app.py")
    print("\nTo start Chatbot:")
    print("  python chatbot/ai_prediction_chatbot.py")


if __name__ == "__main__":
    main()