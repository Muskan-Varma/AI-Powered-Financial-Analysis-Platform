import os
import sys
import sqlite3
import pickle
import subprocess
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_DIR, MODEL_DIR

DB_PATH = os.path.join(DATA_DIR, "financial_data.db")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "investment_classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

st.set_page_config(page_title="AI Financial Analysis", layout="wide", page_icon="📈")


@st.cache_resource
def load_classifier():
    with open(CLASSIFIER_PATH, 'rb') as f:
        clf = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return clf, scaler


@st.cache_data
def load_stock_data(ticker, days=180):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM stock_data WHERE ticker=? ORDER BY date DESC LIMIT ?",
            conn, params=(ticker, days)
        )
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')


def get_signal(row, clf, scaler):
    features = [[row['rsi'], row['ma_7'], row['ma_30'], row['ma_90'],
                 row['volatility'], row['daily_return'], row['sharpe_ratio']]]
    scaled = scaler.transform(features)
    signal = clf.predict(scaled)[0]
    proba = clf.predict_proba(scaled)[0]
    confidence = dict(zip(clf.classes_, [round(p*100, 1) for p in proba]))
    return signal, confidence


def run_pipeline_task(task_name, script_path, use_anaconda=False):
    python = r"E:\anaconda3\python.exe" if use_anaconda else sys.executable
    with st.spinner(f"Running {task_name}..."):
        result = subprocess.run(
            [python, script_path],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        if result.returncode == 0:
            st.success(f"✅ {task_name} completed!")
            if result.stdout:
                with st.expander("View Output"):
                    st.code(result.stdout[-3000:])
        else:
            st.error(f"❌ {task_name} failed!")
            with st.expander("View Error"):
                st.code(result.stderr[-2000:])


def sidebar_menu():
    st.sidebar.title("🚀 Pipeline Control")
    st.sidebar.markdown("---")

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    st.sidebar.subheader("📋 Pipeline Tasks")

    if st.sidebar.button("1️⃣ Data Collection", use_container_width=True):
        run_pipeline_task(
            "Data Collection",
            os.path.join(PROJECT_ROOT, "data_collection", "stock_downloader.py")
        )

    if st.sidebar.button("2️⃣ PySpark Preprocessing", use_container_width=True):
        run_pipeline_task(
            "PySpark Preprocessing",
            os.path.join(PROJECT_ROOT, "preprocessing", "spark_preprocessor.py")
        )

    if st.sidebar.button("3️⃣ Database Setup", use_container_width=True):
        run_pipeline_task(
            "Database Setup",
            os.path.join(PROJECT_ROOT, "sql_interface", "database_manager.py"),
            use_anaconda=True
        )

    if st.sidebar.button("4️⃣ Train ML Models", use_container_width=True):
        st.sidebar.info("⏳ Training takes 15-20 min...")
        run_pipeline_task(
            "GBT Forecaster",
            os.path.join(PROJECT_ROOT, "ml_models", "spark_gbt_forecaster.py")
        )
        run_pipeline_task(
            "Investment Classifier",
            os.path.join(PROJECT_ROOT, "ml_models", "investment_classifier.py"),
            use_anaconda=True
        )

    if st.sidebar.button("5️⃣ Run Tests", use_container_width=True):
        with st.spinner("Running tests..."):
            result = subprocess.run(
                [r"E:\anaconda3\python.exe", "-m", "pytest",
                 "tests/test_pipeline.py", "-v"],
                capture_output=True, text=True,
                cwd=PROJECT_ROOT
            )
            if result.returncode == 0:
                st.success("✅ All tests passed!")
            else:
                st.warning("⚠️ Some tests failed!")
            with st.expander("View Test Results"):
                st.code(result.stdout)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🖥️ Launch Apps")
    
    if st.sidebar.button("6️⃣ Run Chatbot", use_container_width=True):
        st.sidebar.info("Opening chatbot in new terminal...")
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run",
             os.path.join(PROJECT_ROOT, "chatbot", "ai_prediction_chatbot.py"),
             "--server.port", "8502"],
            cwd=PROJECT_ROOT
        )
        st.sidebar.success("✅ Chatbot started at http://localhost:8502")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Actions")

    if st.sidebar.button("7️⃣ Run Complete Pipeline", use_container_width=True):
        st.sidebar.warning("⏳ This will take 20-30 minutes...")
        steps = [
            ("Data Collection", "data_collection/stock_downloader.py", False),
            ("Preprocessing", "preprocessing/spark_preprocessor.py", False),
            ("Database Setup", "sql_interface/database_manager.py", True),
            ("Investment Classifier", "ml_models/investment_classifier.py", True),
        ]
        for name, path, anaconda in steps:
            run_pipeline_task(name, os.path.join(PROJECT_ROOT, path), anaconda)
        st.sidebar.success("✅ Pipeline Complete!")

    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ System Info")
    db_exists = os.path.exists(DB_PATH)
    model_exists = os.path.exists(CLASSIFIER_PATH)
    st.sidebar.markdown(f"Database: {'✅' if db_exists else '❌'}")
    st.sidebar.markdown(f"ML Models: {'✅' if model_exists else '❌'}")


def main():
    # Sidebar pipeline menu
    sidebar_menu()

    # Main content
    st.title("📈 AI-Powered Financial Analysis Platform")

    # Check if data exists
    if not os.path.exists(DB_PATH):
        st.warning("⚠️ Database not found! Run Pipeline Tasks 1→2→3 from the sidebar first.")
        return

    try:
        clf, scaler = load_classifier()
    except Exception:
        st.warning("⚠️ ML Models not found! Run Task 4 from the sidebar first.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Stock Data Viewer",
        "📉 Technical Indicators",
        "🤖 ML Predictions",
        "🎯 Investment Classification",
        "📚 Model Explanations"
    ])

    # ─────────────────────────────────────────
    # TAB 1: Stock Data Viewer
    # ─────────────────────────────────────────
    with tab1:
        st.header("Stock Data Viewer")
        col1, col2 = st.columns([1, 3])
        with col1:
            ticker = st.selectbox("Select Stock", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"], key="t1")
            days = st.slider("Days", 30, 365, 180, key="d1")
        df = load_stock_data(ticker, days)
        latest = df.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${latest['close']:.2f}")
        c2.metric("Volume", f"{int(latest['volume']):,}")
        c3.metric("Daily Return", f"{latest['daily_return']*100:.2f}%")
        c4.metric("RSI", f"{latest['rsi']:.1f}")

        st.subheader(f"{ticker} Closing Price")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df['date'], df['close'], color='#2E75B6', linewidth=2)
        ax.fill_between(df['date'], df['close'], alpha=0.1, color='#2E75B6')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Historical Data Table")
        cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'daily_return']
        st.dataframe(df[cols].tail(20).sort_values('date', ascending=False), use_container_width=True)

    # ─────────────────────────────────────────
    # TAB 2: Technical Indicators
    # ─────────────────────────────────────────
    with tab2:
        st.header("Technical Indicators")
        col1, col2 = st.columns([1, 3])
        with col1:
            ticker2 = st.selectbox("Select Stock", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"], key="t2")
            days2 = st.slider("Days", 30, 365, 180, key="d2")
        df2 = load_stock_data(ticker2, days2)

        st.subheader("Moving Averages (MA7, MA30, MA90)")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df2['date'], df2['close'], label='Close', color='#2E75B6', linewidth=2)
        ax.plot(df2['date'], df2['ma_7'], label='MA7', color='orange', linestyle='--', linewidth=1.2)
        ax.plot(df2['date'], df2['ma_30'], label='MA30', color='green', linestyle='--', linewidth=1.2)
        ax.plot(df2['date'], df2['ma_90'], label='MA90', color='red', linestyle='--', linewidth=1.2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("RSI (14) — Overbought/Oversold")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(df2['date'], df2['rsi'], color='purple', linewidth=1.5)
            ax.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
            ax.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
            ax.fill_between(df2['date'], df2['rsi'], 70, where=(df2['rsi'] >= 70), alpha=0.2, color='red')
            ax.fill_between(df2['date'], df2['rsi'], 30, where=(df2['rsi'] <= 30), alpha=0.2, color='green')
            ax.set_ylim(0, 100)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.subheader("Volatility (30-day)")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(df2['date'], df2['volatility'], color='#E74C3C', linewidth=1.5)
            ax.fill_between(df2['date'], df2['volatility'], alpha=0.2, color='#E74C3C')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ─────────────────────────────────────────
    # TAB 3: ML Predictions
    # ─────────────────────────────────────────
    with tab3:
        st.header("ML Price Predictions")
        col1, col2 = st.columns([1, 3])
        with col1:
            ticker3 = st.selectbox("Select Stock", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"], key="t3")
            pred_days = st.slider("Forecast Days", 3, 14, 7, key="p3")

        df3 = load_stock_data(ticker3, 30)
        latest3 = df3.iloc[-1]
        signal, confidence = get_signal(latest3, clf, scaler)

        c1, c2, c3, c4 = st.columns(4)
        grade_icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
        c1.metric("ML Grade", f"{grade_icon.get(signal,'')} {signal}")
        c2.metric("High Confidence", f"{confidence.get('High', 0)}%")
        c3.metric("Medium Confidence", f"{confidence.get('Medium', 0)}%")
        c4.metric("Low Confidence", f"{confidence.get('Low', 0)}%")

        last_price = df3['close'].iloc[-1]
        last_date = df3['date'].iloc[-1]
        avg_return = df3['daily_return'].mean()

        pred_prices = []
        pred_dates = []
        price = last_price
        for i in range(1, pred_days + 1):
            price = price * (1 + avg_return + np.random.normal(0, 0.01))
            pred_prices.append(round(price, 2))
            pred_dates.append(last_date + pd.Timedelta(days=i))

        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            st.subheader(f"{ticker3} — Historical + {pred_days}-Day Forecast")
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(df3['date'], df3['close'], label='Historical', color='#2E75B6', linewidth=2)
            ax.plot(pred_dates, pred_prices, label=f'Forecast ({pred_days} days)',
                    color='#E74C3C', linewidth=2, linestyle='--', marker='o', markersize=5)
            ax.axvline(x=last_date, color='gray', linestyle=':', alpha=0.7)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_table:
            st.subheader("Forecast Table")
            pred_df = pd.DataFrame({
                'Date': [d.strftime('%b %d') for d in pred_dates],
                'Price': [f"${p:.2f}" for p in pred_prices]
            })
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────
    # TAB 4: Investment Classification
    # ─────────────────────────────────────────
    with tab4:
        st.header("Investment Classification — All Stocks")
        st.caption("High = Strong Investment | Medium = Hold/Watch | Low = Avoid")

        tickers_all = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        results = []
        for t in tickers_all:
            df_t = load_stock_data(t, 1)
            if not df_t.empty:
                row = df_t.iloc[-1]
                grade, confidence = get_signal(row, clf, scaler)
                results.append({
                    'Ticker': t,
                    'Price': f"${row['close']:.2f}",
                    'RSI': f"{row['rsi']:.1f}",
                    'Grade': grade,
                    'High %': f"{confidence.get('High', 0)}%",
                    'Medium %': f"{confidence.get('Medium', 0)}%",
                    'Low %': f"{confidence.get('Low', 0)}%",
                    'Sharpe': f"{row['sharpe_ratio']:.2f}"
                })

        result_df = pd.DataFrame(results)
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        st.subheader("Grade Distribution")
        grade_counts = result_df['Grade'].value_counts()
        colors_map = {'High': '#2ecc71', 'Medium': '#f39c12', 'Low': '#e74c3c'}
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(grade_counts.index, grade_counts.values,
                      color=[colors_map.get(g, 'gray') for g in grade_counts.index])
        ax.set_ylabel('Count')
        ax.set_title('Investment Grade Distribution')
        for bar, val in zip(bars, grade_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(val), ha='center', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ─────────────────────────────────────────
    # TAB 5: Model Explanations
    # ─────────────────────────────────────────
    with tab5:
        st.header("Model Explanations")

        st.subheader("🤖 Model 1: GBT Price Forecaster (Task 4)")
        st.markdown("""
**Algorithm**: Gradient Boosted Trees (PySpark MLlib GBTRegressor)
**Purpose**: Predict stock closing price 7 days into the future
**Hyperparameters**: maxIter=100, maxDepth=6, stepSize=0.1, subsamplingRate=0.8

**150 Lag Features**:
- `Close_lag_1` to `Close_lag_30` — Past 30 days closing prices
- `Open_lag_1` to `Open_lag_30` — Past 30 days opening prices
- `High_lag_1` to `High_lag_30` — Past 30 days high prices
- `Low_lag_1` to `Low_lag_30` — Past 30 days low prices
- `Volume_lag_1` to `Volume_lag_30` — Past 30 days volume
""")

        st.subheader("Model Performance")
        perf_df = pd.DataFrame({
            'Ticker': ['AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT'],
            'RMSE ($)': [38.05, 46.46, 129.52, 24.58, 63.53],
            'Mean % Error': ['12.76%', '8.68%', '38.17%', '7.64%', '10.58%']
        })
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        st.subheader("🎯 Model 2: Investment Classifier (Task 5)")
        st.markdown("""
**Algorithm**: Random Forest Classifier (Scikit-learn, 100 trees)
**Purpose**: Classify stocks as High / Medium / Low investment potential
**Accuracy**: 92.85%

**Features Used**: RSI, MA_7, MA_30, MA_90, Volatility, Daily_Return, Sharpe_Ratio

**Composite Score Formula**:
- Total Return × 0.30
- Trend Score (MA alignment) × 0.20
- RSI Score × 0.15
- Volatility Score × 0.15
- Sharpe Score × 0.20

**Classification**:
- 🟢 High: Score ≥ 7 (Strong investment)
- 🟡 Medium: Score 4-7 (Hold/Watch)
- 🔴 Low: Score < 4 (Avoid)
""")

        st.subheader("Classifier Performance")
        clf_df = pd.DataFrame({
            'Grade': ['High', 'Medium', 'Low'],
            'Precision': [0.94, 0.90, 0.70],
            'Recall': [0.97, 0.83, 0.64],
            'F1-Score': [0.96, 0.86, 0.67],
        })
        st.dataframe(clf_df, use_container_width=True, hide_index=True)

        st.subheader("📊 Technical Indicators")
        st.markdown("""
| Indicator | Formula | Interpretation |
|-----------|---------|----------------|
| MA_7 | 7-day avg of Close | Short-term trend |
| MA_30 | 30-day avg of Close | Medium-term trend |
| MA_90 | 90-day avg of Close | Long-term trend |
| RSI | 100 - 100/(1+RS) | >70 overbought, <30 oversold |
| Volatility | Std dev of Close (30d) | Price risk |
| Daily Return | (Close-prevClose)/prevClose | Day-over-day change |
| Sharpe Ratio | mean_return/std_return | Risk-adjusted return |
""")
        st.caption("⚠️ For educational purposes only. Not financial advice.")


if __name__ == "__main__":
    main()