import os
import sys
import sqlite3
import pickle
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import ollama
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_DIR, MODEL_DIR

DB_PATH = os.path.join(DATA_DIR, "financial_data.db")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "investment_classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


@st.cache_resource
def load_classifier():
    with open(CLASSIFIER_PATH, 'rb') as f:
        clf = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return clf, scaler


def get_stock_data(ticker, days=30):
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM stock_data WHERE ticker=? ORDER BY date DESC LIMIT ?",
            conn, params=(ticker, days)
        )
    return df.sort_values('date')


def get_prediction(ticker, clf, scaler):
    df = get_stock_data(ticker, 1)
    if df.empty:
        return None, None
    row = df.iloc[0]
    features = [[row['rsi'], row['ma_7'], row['ma_30'], row['ma_90'],
                 row['volatility'], row['daily_return'], row['sharpe_ratio']]]
    scaled = scaler.transform(features)
    signal = clf.predict(scaled)[0]
    proba = clf.predict_proba(scaled)[0]
    confidence = dict(zip(clf.classes_, [round(p*100, 1) for p in proba]))
    return signal, confidence


def generate_prediction_graph(ticker):
    df = get_stock_data(ticker, 30)
    if df.empty:
        return None

    last_price = df['close'].iloc[-1]
    last_date = pd.to_datetime(df['date'].iloc[-1])

    # Simple 7-day prediction with slight trend
    daily_return_avg = df['daily_return'].mean()
    predicted_prices = []
    predicted_dates = []
    price = last_price
    for i in range(1, 8):
        price = price * (1 + daily_return_avg + np.random.normal(0, 0.01))
        predicted_prices.append(price)
        predicted_dates.append(last_date + pd.Timedelta(days=i))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(df['date']), df['close'], label='Historical', color='#2E75B6', linewidth=2)
    ax.plot(predicted_dates, predicted_prices, label='Predicted (7 days)', color='#E74C3C',
            linewidth=2, linestyle='--', marker='o', markersize=4)
    ax.axvline(x=last_date, color='gray', linestyle=':', alpha=0.7)
    ax.set_title(f'{ticker} — Last 30 Days + 7-Day Forecast', fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_base64, predicted_prices, predicted_dates


def detect_intent(message):
    msg = message.lower()
    tickers = ['aapl', 'msft', 'googl', 'amzn', 'tsla']
    found_ticker = None
    for t in tickers:
        if t in msg:
            found_ticker = t.upper()
            break

    if any(w in msg for w in ['predict', 'forecast', 'next', 'future', 'price']):
        return 'prediction', found_ticker
    elif any(w in msg for w in ['show', 'data', 'history', 'tell me about', 'info']):
        return 'data', found_ticker
    elif found_ticker:
        return 'prediction', found_ticker
    else:
        return 'chat', None


def get_llm_response(message, context=""):
    system = "You are an AI financial analysis assistant. Be concise and clear. Always note this is not financial advice."
    if context:
        system += f"\n\nStock context:\n{context}"
    response = ollama.chat(
        model='llama3.2',
        messages=[{"role": "user", "content": message}],
        options={"system": system}
    )
    return response['message']['content']


def main():
    st.set_page_config(page_title="AI Stock Chatbot", page_icon="🤖", layout="wide")
    st.title("🤖 AI Financial Prediction Chatbot")
    st.caption("Ask me about AAPL, MSFT, GOOGL, AMZN, TSLA — predictions, data, or concepts!")

    clf, scaler = load_classifier()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "image" in msg:
                st.image(base64.b64decode(msg["image"]))
            if "dataframe" in msg:
                st.dataframe(msg["dataframe"], use_container_width=True)

    # Chat input
    user_input = st.chat_input("Ask me anything... e.g. 'Predict AAPL next 7 days'")

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        intent, ticker = detect_intent(user_input)

        with st.chat_message("assistant"):
            if intent == 'prediction' and ticker:
                signal, confidence = get_prediction(ticker, clf, scaler)
                result = generate_prediction_graph(ticker)

                if result and result[0]:
                    img_b64, pred_prices, pred_dates = result
                    text = f"**{ticker} Analysis**\n\n"
                    text += f"📊 **ML Signal**: {signal}\n"
                    text += f"🎯 **Confidence**: BUY {confidence.get('BUY',0)}% | HOLD {confidence.get('HOLD',0)}% | SELL {confidence.get('SELL',0)}%\n\n"
                    text += f"📈 **7-Day Price Forecast**:\n"
                    for i, (d, p) in enumerate(zip(pred_dates, pred_prices)):
                        text += f"  Day {i+1} ({d.strftime('%b %d')}): **${p:.2f}**\n"
                    text += "\n⚠️ *Not financial advice — educational purposes only.*"

                    st.markdown(text)
                    st.image(base64.b64decode(img_b64))
                    st.session_state.messages.append({
                        "role": "assistant", "content": text, "image": img_b64
                    })

            elif intent == 'data' and ticker:
                df = get_stock_data(ticker, 10)
                display = df[['date', 'close', 'rsi', 'ma_7', 'ma_30', 'daily_return']].copy()
                display.columns = ['Date', 'Close', 'RSI', 'MA7', 'MA30', 'Daily Return']
                text = f"**{ticker} — Last 10 Days Data**"
                st.markdown(text)
                st.dataframe(display, use_container_width=True)
                st.session_state.messages.append({
                    "role": "assistant", "content": text,
                    "dataframe": display
                })

            else:
                response = get_llm_response(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()