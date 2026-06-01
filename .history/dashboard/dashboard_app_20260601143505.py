import os
import sys
import sqlite3
import pickle
import pandas as pd
import numpy as np
import streamlit as st
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
def load_all_latest():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("""
            SELECT * FROM stock_data
            WHERE date = (SELECT MAX(date) FROM stock_data WHERE ticker = s.ticker)
            FROM stock_data s
            GROUP BY ticker
        """, conn)
    return df


def get_signal(row, clf, scaler):
    features = [[row['rsi'], row['ma_7'], row['ma_30'], row['ma_90'],
                 row['volatility'], row['daily_return'], row['sharpe_ratio']]]
    scaled = scaler.transform(features)
    signal = clf.predict(scaled)[0]
    proba = clf.predict_proba(scaled)[0]
    confidence = dict(zip(clf.classes_, [round(p*100, 1) for p in proba]))
    return signal, confidence


def main():
    st.title("📈 AI-Powered Financial Analysis Platform")
    st.markdown("---")

    clf, scaler = load_classifier()

    # Sidebar
    st.sidebar.title("Settings")
    ticker = st.sidebar.selectbox("Select Stock", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])
    days = st.sidebar.slider("Days of History", 30, 365, 180)

    df = load_stock_data(ticker, days)

    if df.empty:
        st.error("No data found!")
        return

    latest = df.iloc[-1]
    signal, confidence = get_signal(latest, clf, scaler)

    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Price", f"${latest['close']:.2f}")
    col2.metric("RSI", f"{latest['rsi']:.1f}")
    col3.metric("Daily Return", f"{latest['daily_return']*100:.2f}%")
    col4.metric("Sharpe Ratio", f"{latest['sharpe_ratio']:.2f}")

    signal_color = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}
    col5.metric("ML Signal", f"{signal_color.get(signal,'')} {signal}")

    st.markdown("---")

    # Price chart
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader(f"{ticker} Price & Moving Averages")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['date'], df['close'], label='Close', linewidth=2, color='#1f77b4')
        ax.plot(df['date'], df['ma_7'], label='MA7', linewidth=1, linestyle='--', color='orange')
        ax.plot(df['date'], df['ma_30'], label='MA30', linewidth=1, linestyle='--', color='green')
        ax.plot(df['date'], df['ma_90'], label='MA90', linewidth=1, linestyle='--', color='red')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Signal Confidence")
        labels = list(confidence.keys())
        values = list(confidence.values())
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title(f"Signal: {signal}")
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")

    # RSI + Volume
    col3a, col3b = st.columns(2)

    with col3a:
        st.subheader("RSI (14)")
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        ax3.plot(df['date'], df['rsi'], color='purple', linewidth=1.5)
        ax3.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax3.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax3.set_ylim(0, 100)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col3b:
        st.subheader("Volume")
        fig4, ax4 = plt.subplots(figsize=(6, 3))
        ax4.bar(df['date'], df['volume'], color='#3498db', alpha=0.7, width=1)
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.markdown("---")
    st.subheader("Recent Data")
    display_cols = ['date', 'close', 'rsi', 'ma_7', 'ma_30', 'daily_return', 'sharpe_ratio']
    st.dataframe(df[display_cols].tail(10).sort_values('date', ascending=False), use_container_width=True)

    st.caption("⚠️ This is not financial advice. For educational purposes only.")


if __name__ == "__main__":
    main()