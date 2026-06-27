"""
data_prep.py
------------
Phase 2 - Data preparation and RFM feature engineering.
Reads raw Online Retail data, cleans it, computes RFM features,
applies log1p transform and RobustScaler, and saves the feature
store to data/processed/rfm_processed.csv.
Also exports the fitted scaler to models/scaler.pkl for use by the API.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


def load_data(path: str) -> pd.DataFrame:
    """Load raw Online Retail CSV file into a DataFrame."""
    df = pd.read_csv(path, dtype={"CustomerID": str})
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw transaction data by removing:
    - Cancelled orders (InvoiceNo starts with 'C')
    - Rows with missing CustomerID
    - Exact duplicate rows
    - Rows with Quantity <= 0 or UnitPrice <= 0
    """
    before = len(df)

    # Remove cancelled orders
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove missing CustomerIDs
    df = df.dropna(subset=["CustomerID"])

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Remove invalid Quantity and UnitPrice
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    after = len(df)
    print(f"Cleaned data: {before:,} → {after:,} rows ({before - after:,} removed)")
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM features per customer.
    - Recency: days since last purchase (lower = more recent)
    - Frequency: number of distinct invoices
    - Monetary: total spend (Quantity x UnitPrice)
    Reference date: 2012-01-01 (one day after last transaction)
    """
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    reference_date = pd.Timestamp("2012-01-01")

    df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalSpend", "sum")
    ).reset_index()

    print(f"RFM computed for {len(rfm):,} customers")
    return rfm


def transform_and_scale(rfm: pd.DataFrame) -> tuple:
    """
    Apply log1p transform to reduce right skew, then scale
    using RobustScaler (robust to outliers in spend data).
    Returns a tuple of (scaled DataFrame, fitted scaler).
    The fitted scaler must be saved and used by the API to
    transform live incoming data to match the model training space.
    """
    rfm_transformed = rfm.copy()

    # log1p transform (log(1 + x)) — handles skew safely even if value is 0
    rfm_transformed["Recency"] = np.log1p(rfm_transformed["Recency"])
    rfm_transformed["Frequency"] = np.log1p(rfm_transformed["Frequency"])
    rfm_transformed["Monetary"] = np.log1p(rfm_transformed["Monetary"])

    # RobustScaler — scales using median and IQR, not mean/std
    scaler = RobustScaler()
    rfm_transformed[["Recency", "Frequency", "Monetary"]] = scaler.fit_transform(
        rfm_transformed[["Recency", "Frequency", "Monetary"]]
    )

    print("log1p transform and RobustScaler applied")
    return rfm_transformed, scaler


def save_scaler(scaler: RobustScaler, path: str) -> None:
    """
    Save the fitted RobustScaler to disk using pickle.
    Required by the Phase 5 prediction API to transform
    live incoming RFM values before passing to the model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {path}")


def save_feature_store(df: pd.DataFrame, path: str) -> None:
    """Save the processed RFM DataFrame to the feature store path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Feature store saved to {path}")


if __name__ == "__main__":
    RAW_PATH = "data/raw/online_retail.csv"
    PROCESSED_PATH = "data/processed/rfm_processed.csv"
    SCALER_PATH = "models/scaler.pkl"

    df = load_data(RAW_PATH)
    df = clean_data(df)
    rfm = compute_rfm(df)
    rfm_scaled, scaler = transform_and_scale(rfm)
    save_scaler(scaler, SCALER_PATH)
    save_feature_store(rfm_scaled, PROCESSED_PATH)