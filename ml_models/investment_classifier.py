import os
import sys
import pandas as pd
import numpy as np
import sqlite3
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_DIR, MODEL_DIR

DB_PATH = os.path.join(DATA_DIR, "financial_data.db")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "investment_classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


class InvestmentClassifier:

    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.feature_cols = ['rsi', 'ma_7', 'ma_30', 'ma_90', 'volatility', 'daily_return', 'sharpe_ratio']

    def load_data(self):
        print("Loading data from SQLite...")
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query("SELECT * FROM stock_data ORDER BY ticker, date", conn)
        print(f"  Loaded {len(df)} rows")
        return df

    def create_labels(self, df):
        print("Creating Buy/Hold/Sell labels...")
        df = df.copy()
        df['future_return'] = df.groupby('ticker')['close'].transform(
            lambda x: x.shift(-7) / x - 1
        )
        def label(r):
            if r > 0.03:
                return 'BUY'
            elif r < -0.03:
                return 'SELL'
            else:
                return 'HOLD'
        df['label'] = df['future_return'].apply(label)
        df = df.dropna(subset=['future_return'] + self.feature_cols)
        counts = df['label'].value_counts()
        print(f"  BUY: {counts.get('BUY',0)} | HOLD: {counts.get('HOLD',0)} | SELL: {counts.get('SELL',0)}")
        return df

    def prepare_features(self, df):
        X = df[self.feature_cols].values
        y = df['label'].values
        return X, y

    def train(self, X_train, y_train):
        print("Training Random Forest classifier...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("  Training complete!")

    def evaluate(self, X_test, y_test):
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        return acc

    def save_model(self):
        with open(CLASSIFIER_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  Model saved: {CLASSIFIER_PATH}")

    def load_model(self):
        with open(CLASSIFIER_PATH, 'rb') as f:
            self.model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)

    def predict(self, ticker, rsi, ma_7, ma_30, ma_90, volatility, daily_return, sharpe_ratio):
        features = np.array([[rsi, ma_7, ma_30, ma_90, volatility, daily_return, sharpe_ratio]])
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        proba = self.model.predict_proba(features_scaled)[0]
        classes = self.model.classes_
        confidence = dict(zip(classes, [round(p*100, 1) for p in proba]))
        return prediction, confidence


def run_classification():
    print("="*60)
    print("INVESTMENT CLASSIFIER (Buy/Hold/Sell)")
    print("="*60)
    clf = InvestmentClassifier()
    df = clf.load_data()
    df = clf.create_labels(df)
    X, y = clf.prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.train(X_train, y_train)
    clf.evaluate(X_test, y_test)
    clf.save_model()

    print("\nSample prediction (AAPL latest):")
    with sqlite3.connect(DB_PATH) as conn:
        row = pd.read_sql_query(
            "SELECT * FROM stock_data WHERE ticker='AAPL' ORDER BY date DESC LIMIT 1", conn
        ).iloc[0]
    signal, confidence = clf.predict(
        'AAPL', row['rsi'], row['ma_7'], row['ma_30'], row['ma_90'],
        row['volatility'], row['daily_return'], row['sharpe_ratio']
    )
    print(f"  Signal: {signal}")
    print(f"  Confidence: {confidence}")
    return clf


if __name__ == "__main__":
    run_classification()