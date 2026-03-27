"""
Fetches Walmart stock prices and Consumer Confidence Index from public sources.
"""

import yfinance as yf
import pandas as pd

def fetch_walmart_stock(start="2012-01-01", end="2023-12-31"):
    """Fetch weekly Walmart (WMT) closing prices from Yahoo Finance."""
    wmt = yf.Ticker("WMT")
    weekly_prices = wmt.history(start=start, end=end, interval="1wk")[["Close"]]
    weekly_prices.index = pd.to_datetime(weekly_prices.index).tz_localize(None)
    return weekly_prices


def fetch_consumer_confidence(filepath="data/consumer_confidence.csv"):
    """Load and resample monthly Consumer Confidence Index (FRED) to weekly."""
    cci = pd.read_csv(filepath, parse_dates=["DATE"], index_col="DATE")
    cci.columns = ["consumer_confidence"]
    return cci.resample("W").interpolate(method="linear")

if __name__ == "__main__":
    prices = fetch_walmart_stock()
    prices.to_csv("data/wmt_stock.csv")
    print(prices.tail())