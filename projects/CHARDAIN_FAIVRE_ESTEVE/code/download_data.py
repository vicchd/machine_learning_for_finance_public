"""
Downloads the PKLot parking dataset from Kaggle.
"""

import kagglehub

def download_pklot():
    """Download PKLot dataset via kagglehub."""
    path = kagglehub.dataset_download("ammarmahali/pklot-dataset")
    print(f"Dataset downloaded to: {path}")
    return path

if __name__ == "__main__":
    download_pklot()
