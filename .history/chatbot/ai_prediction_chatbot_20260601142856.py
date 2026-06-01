import os
import sys
import sqlite3
import pandas as pd
import pickle
import ollama

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_DIR, MODEL_DIR

DB_PATH = os.path.join(DATA_DIR, "financial_data.db")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "investment_classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


class AIPredictionChatbot:

    def __init__(self):
        self.model = "llama3.2"
        self.conversation_history = []
        self._load_classifier()

    def _load_classifier(self):
        try:
            with open(CLASSIFIER_PATH, 'rb') as f:
                self.classifier = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            print("  Classifier loaded!")
        except Exception as e:
            print(f"  Classifier not loaded: {e}")
            self.classifier = None

    def get_stock_info(self, ticker):
        ticker = ticker.upper()
        try:
            with sqlite3.connect(DB_PATH) as conn:
                df = pd.read_sql_query(
                    "SELECT * FROM stock_data WHERE ticker=? ORDER BY date DESC LIMIT 30",
                    conn, params=(ticker,)
                )
            if df.empty:
                return None
            return df
        except:
            return None

    def get_signal(self, ticker):
        df = self.get_stock_info(ticker)
        if df is None or self.classifier is None:
            return None, None, None
        row = df.iloc[0]
        features = [[row['rsi'], row['ma_7'], row['ma_30'], row['ma_90'],
                     row['volatility'], row['daily_return'], row['sharpe_ratio']]]
        import numpy as np
        features_scaled = self.scaler.transform(features)
        signal = self.classifier.predict(features_scaled)[0]
        proba = self.classifier.predict_proba(features_scaled)[0]
        confidence = dict(zip(self.classifier.classes_, [round(p*100,1) for p in proba]))
        return signal, confidence, row

    def build_context(self, ticker):
        signal, confidence, row = self.get_signal(ticker)
        if signal is None:
            return ""
        context = f"""
Stock: {ticker}
Date: {row['date']}
Current Price: ${row['close']:.2f}
RSI: {row['rsi']:.2f}
MA7: ${row['ma_7']:.2f} | MA30: ${row['ma_30']:.2f} | MA90: ${row['ma_90']:.2f}
Volatility: {row['volatility']:.2f}
Daily Return: {row['daily_return']*100:.2f}%
Sharpe Ratio: {row['sharpe_ratio']:.2f}
ML Signal: {signal}
Confidence: BUY={confidence.get('BUY',0)}% | HOLD={confidence.get('HOLD',0)}% | SELL={confidence.get('SELL',0)}%
"""
        return context

    def detect_ticker(self, message):
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        msg_upper = message.upper()
        for t in tickers:
            if t in msg_upper:
                return t
        return None

    def chat(self, user_message):
        ticker = self.detect_ticker(user_message)
        system_prompt = """You are an AI financial analysis assistant. 
You help users understand stock data and ML predictions.
Always remind users that this is not financial advice.
Be concise and clear."""

        if ticker:
            context = self.build_context(ticker)
            if context:
                system_prompt += f"\n\nCurrent stock data:\n{context}"

        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        messages = [{"role": "system", "content": system_prompt}] + self.conversation_history

        response = ollama.chat(
            model=self.model,
            messages=messages
        )

        reply = response['message']['content']
        self.conversation_history.append({
            "role": "assistant",
            "content": reply
        })
        return reply

    def run(self):
        print("="*60)
        print("AI FINANCIAL CHATBOT (LLaMA 3.2)")
        print("="*60)
        print("Ask about: AAPL, MSFT, GOOGL, AMZN, TSLA")
        print("Type 'quit' to exit\n")
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            print("\nBot: ", end="", flush=True)
            reply = self.chat(user_input)
            print(reply)
            print()


if __name__ == "__main__":
    chatbot = AIPredictionChatbot()
    chatbot.run()