"""
Main Pipeline Orchestrator - AI Financial Analysis Platform
Usage: python main.py
"""

import os
import sys
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("="*60)
    print("   AI-POWERED FINANCIAL ANALYSIS PLATFORM")
    print("="*60)
    print("\nStarting Dashboard...")

    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dashboard", "dashboard_app.py"
    )

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", dashboard_path
    ])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()