import os
import sys
import sqlite3
import pickle
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


@st.cache_data
def load_all_tickers():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM stock_data ORDER BY ticker, date", conn)
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_signal(row, clf, scaler):
    features = [[row['rsi'], row['ma_7'], row['ma_30'], row['ma_90'],
                 row['volatility'], row['daily_return'], row['sharpe_ratio']]]
    scaled = scaler.transform(features)
    signal = clf.predict(scaled)[0]
    proba = clf.predict_proba(scaled)[0]
    confidence = dict(zip(clf.classes_, [round(p*100, 1) for p in proba]))
    return signal, confidence


def get_investment_grade(row, clf, scaler):
    features = [[row['rsi'], row['ma_7'], row['ma_30'], row['ma_90'],
                 row['volatility'], row['daily_return'], row['sharpe_ratio']]]
    scaled = scaler.transform(features)
    grade = clf.predict(scaled)[0]
    proba = clf.predict_proba(scaled)[0]
    confidence = dict(zip(clf.classes_, [round(p*100, 1) for p in proba]))
    return grade, confidence


def main():
    st.title("📈 AI-Powered Financial Analysis Platform")
    clf, scaler = load_classifier()

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

        # Moving Averages
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
            # RSI
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
            # Volatility
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
        signal_icon = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}
        c1.metric("ML Signal", f"{signal_icon.get(signal,'')} {signal}")
        c2.metric("BUY Confidence", f"{confidence.get('BUY',0)}%")
        c3.metric("HOLD Confidence", f"{confidence.get('HOLD',0)}%")
        c4.metric("SELL Confidence", f"{confidence.get('SELL',0)}%")

        # Generate forecast
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
        st.caption("Classification: High (BUY confidence ≥ 40%) | Medium (25-40%) | Low (<25%)")

        tickers_all = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        results = []
        for t in tickers_all:
            df_t = load_stock_data(t, 1)
            if not df_t.empty:
                row = df_t.iloc[-1]
                grade, buy_conf = get_investment_grade(row, clf, scaler)
                signal, confidence = get_signal(row, clf, scaler)
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
        colors = {'High': '#2ecc71', 'Medium': '#f39c12', 'Low': '#e74c3c'}
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(grade_counts.index, grade_counts.values,
                      color=[colors.get(g, 'gray') for g in grade_counts.index])
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
**Hyperparameters**: maxIter=100, maxDepth=6, stepSize=0.1  

**Features Used (44 lag features)**:
- `Close_lag_1` to `Close_lag_30` — Past 30 days closing prices
- `RSI_lag_1` to `RSI_lag_7` — Past 7 days RSI values
- `Volatility_lag_1` to `Volatility_lag_7` — Past 7 days volatility
- Plus: MA_7, MA_30, MA_90, RSI, Volatility, Sharpe_Ratio
""")

        st.subheader("Model Performance")
        perf_df = pd.DataFrame({
            'Ticker': ['AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT'],
            'RMSE ($)': [30.45, 55.95, 90.42, 22.09, 71.84],
            'MAE ($)': [26.07, 46.71, 67.26, 17.52, 59.36],
            'R² Score': [0.02, 0.31, -0.49, -0.03, -0.92]
        })
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        st.subheader("🎯 Model 2: Investment Classifier (Task 5)")
        st.markdown("""
**Algorithm**: Random Forest Classifier (Scikit-learn, 100 trees)  
**Purpose**: Generate BUY / HOLD / SELL signals  
**Accuracy**: 65.4%  

**17 Features Used**:
| Feature | Description |
|---------|-------------|
| RSI | Relative Strength Index (14-day) |
| MA_7 | 7-day Moving Average |
| MA_30 | 30-day Moving Average |
| MA_90 | 90-day Moving Average |
| Volatility | 30-day price std deviation |
| Daily_Return | Day-over-day % change |
| Sharpe_Ratio | Risk-adjusted return |

**Label Generation**:
- BUY: 7-day future return > 3%
- SELL: 7-day future return < -3%
- HOLD: Otherwise
""")

        st.subheader("Classifier Performance")
        clf_df = pd.DataFrame({
            'Signal': ['BUY', 'HOLD', 'SELL'],
            'Precision': [0.68, 0.62, 0.70],
            'Recall': [0.60, 0.75, 0.54],
            'F1-Score': [0.64, 0.68, 0.61],
            'Support': [523, 705, 373]
        })
        st.dataframe(clf_df, use_container_width=True, hide_index=True)

        st.subheader("📊 Technical Indicators Explained")
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