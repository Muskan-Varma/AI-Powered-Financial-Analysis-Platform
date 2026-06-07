"""
Main Pipeline Orchestrator - AI Financial Analysis Platform
Usage: python main.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def initialize_spark():
    from pyspark.sql import SparkSession
    print("\nInitializing Spark session...")
    spark = SparkSession.builder \
        .appName("FinancialAnalysisPlatform") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.debug.maxToStringFields", "200") \
        .getOrCreate()
    print("Spark session ready!")
    return spark


def run_data_collection():
    from data_collection.stock_downloader import run_data_collection as rdc
    rdc()


def run_preprocessing(spark):
    from preprocessing.spark_preprocessor import run_preprocessing as rp
    rp(spark)


def run_database():
    from sql_interface.database_manager import run_database_setup
    run_database_setup()


def run_ml_models(spark):
    from ml_models.spark_gbt_forecaster import run_forecasting
    from ml_models.investment_classifier import run_classification
    run_forecasting(spark)
    run_classification()


def run_complete_pipeline():
    print("\n" + "="*60)
    print("RUNNING COMPLETE PIPELINE")
    print("="*60)
    spark = initialize_spark()
    run_data_collection()
    run_preprocessing(spark)
    run_database()
    run_ml_models(spark)
    spark.stop()
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nTo start Dashboard: streamlit run dashboard/dashboard_app.py")
    print("To start Chatbot:   streamlit run chatbot/ai_prediction_chatbot.py")


def print_menu():
    print("\n" + "="*60)
    print("   AI-POWERED FINANCIAL ANALYSIS PLATFORM")
    print("="*60)
    print("  1. Data Collection")
    print("  2. PySpark Preprocessing")
    print("  3. Database Setup")
    print("  4. Train ML Models")
    print("  5. Run Chatbot")
    print("  6. Run Dashboard")
    print("  7. Run Complete Pipeline")
    print("  0. Exit")
    print("="*60)


def main():
    spark = None
    while True:
        print_menu()
        choice = input("\nSelect option (0-7): ").strip()

        if choice == '0':
            print("Goodbye!")
            if spark:
                spark.stop()
            break

        elif choice == '1':
            run_data_collection()

        elif choice == '2':
            if not spark:
                spark = initialize_spark()
            run_preprocessing(spark)

        elif choice == '3':
            run_database()

        elif choice == '4':
            if not spark:
                spark = initialize_spark()
            run_ml_models(spark)

        elif choice == '5':
            print("\nStarting Chatbot...")
            os.system("streamlit run chatbot/ai_prediction_chatbot.py")

        elif choice == '6':
            print("\nStarting Dashboard...")
            os.system("streamlit run dashboard/dashboard_app.py")

        elif choice == '7':
            run_complete_pipeline()

        else:
            print("Invalid option! Please select 0-7.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()