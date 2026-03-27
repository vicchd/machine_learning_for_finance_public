"""
Extracts parking occupancy rate from labeled images using COCO annotations.
"""

import json
import os
from datetime import datetime
import pandas as pd


def parse_date_from_filename(filename):
    """Extract date from PKLot filename: YYYY-MM-DD_HH_MM_SS_jpg..."""
    try:
        return datetime.strptime(filename[:10], "%Y-%m-%d")
    except ValueError:
        return None


def compute_occupancy(annotations_path):
    """
    Compute daily parking occupancy rate from COCO annotations.
    Category 1 = occupied, Category 0 = empty 
    Returns a DataFrame with columns: date, occupancy_rate.
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    images = {img["id"]: img["file_name"] for img in coco["images"]}

    records = []
    for ann in coco["annotations"]:
        filename = images[ann["image_id"]]
        date = parse_date_from_filename(filename)
        label = categories[ann["category_id"]]
        if date:
            records.append({"date": date, "occupied": 1 if label == "space-occupied" else 0})

    df = pd.DataFrame(records)
    daily = df.groupby("date")["occupied"].mean().reset_index()
    daily.columns = ["date", "occupancy_rate"]
    return daily.set_index("date")


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        path = f"data/images/{split}/_annotations.coco.json"
        if os.path.exists(path):
            df = compute_occupancy(path)
            print(f"\n{split}:")
            print(df.head())