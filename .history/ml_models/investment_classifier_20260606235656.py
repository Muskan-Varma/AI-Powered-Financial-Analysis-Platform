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

    def engineer_features(self, df):
        print("Engineering features...")
        df = df.copy()

        # Per ticker aggregate features
        result = []
        for ticker in df['ticker'].unique():
            t_df = df[df['ticker'] == ticker].copy().sort_values('date')

            # Total return
            t_df['total_return'] = (t_df['close'] - t_df['close'].iloc[0]) / t_df['close'].iloc[0]

            # 7-day and 30-day recent returns
            t_df['return_7d'] = t_df['close'].pct_change(7)
            t_df['return_30d'] = t_df['close'].pct_change(30)

            # MA trend signals
            t_df['ma7_vs_ma30'] = (t_df['ma_7'] - t_df['ma_30']) / t_df['ma_30']
            t_df['ma30_vs_ma90'] = (t_df['ma_30'] - t_df['ma_90']) / t_df['ma_90']

            # Average RSI
            t_df['avg_rsi'] = t_df['rsi'].rolling(14).mean()

            result.append(t_df)

        df = pd.concat(result)
        return df

    def calculate_composite_score(self, row):
        # Normalize each component to 0-10 scale
        # Total return score (0-10)
        total_return_score = min(max(row['total_return'] * 100, 0), 10)

        # Trend score based on MA alignment
        trend_score = 5.0
        if row['ma7_vs_ma30'] > 0 and row['ma30_vs_ma90'] > 0:
            trend_score = 8.0
        elif row['ma7_vs_ma30'] < 0 and row['ma30_vs_ma90'] < 0:
            trend_score = 2.0

        # RSI score (ideal RSI = 50, penalize extremes)
        rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
        rsi_score = 10 - abs(rsi - 50) / 5

        # Volatility score (lower volatility = better, max 20 std)
        vol = row['volatility'] if not pd.isna(row['volatility']) else 10
        volatility_score = max(10 - vol / 5, 0)

        # Sharpe score
        sharpe = row['sharpe_ratio'] if not pd.isna(row['sharpe_ratio']) else 0
        sharpe_score = min(max((sharpe + 1) * 5, 0), 10)

        # Composite score (weighted)
        score = (total_return_score * 0.30 +
                 trend_score * 0.20 +
                 rsi_score * 0.15 +
                 volatility_score * 0.15 +
                 sharpe_score * 0.20)
        return round(score, 2)

    def create_labels(self, df):
        print("Creating High/Medium/Low labels...")
        df = df.copy()
        df['composite_score'] = df.apply(self.calculate_composite_score, axis=1)

        def label(score):
            if score >= 7:
                return 'High'
            elif score >= 4:
                return 'Medium'
            else:
                return 'Low'

        df['label'] = df['composite_score'].apply(label)
        df = df.dropna(subset=self.feature_cols + ['label'])
        counts = df['label'].value_counts()
        print(f"  High: {counts.get('High',0)} | Medium: {counts.get('Medium',0)} | Low: {counts.get('Low',0)}")
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

    def predict(self, features):
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        proba = self.model.predict_proba(features_scaled)[0]
        confidence = dict(zip(self.model.classes_, [round(p*100, 1) for p in proba]))
        return prediction, confidence

    def classify_all_tickers(self, df):
        print("\nClassification Results:")
        print("="*50)
        for ticker in df['ticker'].unique():
            t_df = df[df['ticker'] == ticker].sort_values('date')
            if t_df.empty:
                continue
            row = t_df.iloc[-1]
            features = [row[c] for c in self.feature_cols]
            if any(pd.isna(f) for f in features):
                continue
            grade, confidence = self.predict(features)
            score = row.get('composite_score', 0)
            print(f"  {ticker}: {grade} (Score: {score:.1f}) — Confidence: {confidence}")


def run_classification():
    print("="*60)
    print("INVESTMENT CLASSIFIER (High/Medium/Low)")
    print("="*60)
    clf = InvestmentClassifier()
    df = clf.load_data()
    df = clf.engineer_features(df)
    df = clf.create_labels(df)
    X, y = clf.prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.train(X_train, y_train)
    clf.evaluate(X_test, y_test)
    clf.save_model()
    clf.classify_all_tickers(df)
    return clf


if __name__ == "__main__":
    run_classification()