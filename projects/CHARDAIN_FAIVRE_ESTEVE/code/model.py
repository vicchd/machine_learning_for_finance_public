"""
Trains and evaluates ML models to predict Walmart stock direction from parking lot occupancy rates.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def load_data():
    """Load and split the merged dataset."""
    df = pd.read_csv("data/dataset.csv", index_col=0, parse_dates=True)    
    X = df[["occupancy_rate"]]
    y = df["price_direction"]
    return train_test_split(X, y, test_size=0.2, shuffle=False)


def evaluate(model, X_test, y_test, name):
    """Print accuracy and classification report for a given model."""
    preds = model.predict(X_test)
    print(f"\n{name} — Accuracy: {accuracy_score(y_test, preds):.2f}")
    print(classification_report(y_test, preds, zero_division=0))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    # Naive benchmark: always predict majority class
    majority = y_train.mode()[0]
    naive_preds = [majority] * len(y_test)
    print(f"Naive benchmark — Accuracy: {accuracy_score(y_test, naive_preds):.2f}")

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    evaluate(lr, X_test, y_test, "Logistic Regression")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    evaluate(rf, X_test, y_test, "Random Forest")