"""
Main Pipeline Orchestrator - AI Financial Analysis Platform

STUDENT TASK:
Complete this main pipeline file to orchestrate all components

USAGE:
    python main.py

Then select from menu:
1. Data Collection
2. Preprocessing
3. Database Setup
4. Train ML Models
5. Run Chatbot
6. Run Dashboard
7. Run Complete Pipeline
"""

import os
import sys
from pyspark.sql import SparkSession

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import *


def initialize_spark():
    """
    Initialize Spark session

    TODO: Students - Create Spark session with proper configuration
    """
    print("\nInitializing Spark session...")


    print("\nTODO: Complete the menu logic in main()")


if __name__ == "__main__":
    """
    Entry point

    Usage: python main.py
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
