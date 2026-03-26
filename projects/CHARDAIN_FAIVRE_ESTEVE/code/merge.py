"""
Merges parking occupancy rates with Walmart stock prices.
"""

import pandas as pd
from features import compute_occupancy

def build_dataset():
    """Merge occupancy data from all splits with Walmart weekly closing prices."""
    splits = ["train", "valid", "test"]
    occupancy = pd.concat([
        compute_occupancy(f"data/images/{s}/_annotations.coco.json")
        for s in splits
    ])
    occupancy = occupancy.groupby("date").mean()
    occupancy.index = pd.to_datetime(occupancy.index)

    prices = pd.read_csv("data/wmt_stock.csv", parse_dates=["Date"], index_col="Date")
    prices = prices.resample("D").interpolate(method="linear")

    df = occupancy.join(prices, how="inner")
    df["price_direction"] = (df["Close"].diff() > 0).astype(int)
    df = df.dropna()
    return df


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("data/dataset.csv")
    print(df.head(10))
    print(f"\nTotal samples: {len(df)}")