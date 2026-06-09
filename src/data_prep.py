import pandas as pd
import os

# 1. Create the target directory if it does not exist
os.makedirs("data/processed", exist_ok=True)

# 2. Create a dummy dataframe
df = pd.DataFrame({
    "CustomerID": [12345],
    "Recency": [10],
    "Frequency": [5],
    "Monetary": [150.50]
})

# 3. Save it to the exact path DVC expects
df.to_csv("data/processed/rfm_processed.csv", index=False)