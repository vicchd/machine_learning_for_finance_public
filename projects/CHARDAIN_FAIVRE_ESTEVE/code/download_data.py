"""
Downloads the PKLot parking dataset from Kaggle.
"""

import kagglehub
import shutil
import os

def download_pklot():
    """Download PKLot dataset via kagglehub and move it to data/images/."""
    path = kagglehub.dataset_download("ammarmahali/pklot-dataset")
    dest = "data/images"
    if not os.path.exists(dest):
        shutil.copytree(path, dest)
        print(f"Dataset moved to: {dest}")
    else:
        print("data/images already exists, skipping copy.")
    return dest

