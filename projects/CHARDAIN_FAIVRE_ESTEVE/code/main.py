"""
The main script runs the full pipeline: data collection, feature extraction, merging, and modeling.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_collection import fetch_walmart_stock
from merge import build_dataset
from model import evaluate

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=== First step: Fetching Walmart stock data ===")
    prices = fetch_walmart_stock()
    prices.to_csv("data/wmt_stock.csv")

    print("\n=== Now, we are building the dataset ===")
    df = build_dataset()
    df.to_csv("data/dataset.csv")
    print(f"Dataset shape: {df.shape}")

    print("\n=== Third step: Training models ===")
    X = df[["occupancy_rate"]]
    y = df["price_direction"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    majority = y_train.mode()[0]
    naive_acc = accuracy_score(y_test, [majority] * len(y_test))
    print(f"Naive benchmark accuracy: {naive_acc:.2f}")

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    evaluate(lr, X_test, y_test, "Logistic Regression")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    evaluate(rf, X_test, y_test, "Random Forest")

    print("\n=== All steps completed ===")


if __name__ == "__main__":
    main()