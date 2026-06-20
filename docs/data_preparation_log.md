# Data Preparation Log — `src/data_prep.py`

**Author:** Ayush | **Phase:** 2 — Data & EDA | **Date:** June 2026

## Objective
Transform raw UCI Online Retail transaction data into a clean, scaled RFM
feature store to support customer segmentation modelling.

## 1. Input
- Source: UCI Online Retail Dataset
- Rows: 541,909 transactions
- Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
- Location: `data/raw/online_retail.csv` (DVC tracked)

## 2. Cleaning Steps (in order)
| Step | Rule | Rationale |
|---|---|---|
| Remove cancellations | `InvoiceNo` starts with `'C'` | Cancelled orders don't reflect real purchase behaviour |
| Remove null CustomerID | Drop rows with missing ID | Cannot build a customer profile without an identifier (~25% of rows) |
| Remove duplicates | Exact row matches | Avoid double-counting transactions |
| Remove invalid values | `Quantity <= 0` or `UnitPrice <= 0` | Returns/data errors, not valid purchases |

**Result:** 541,909 → 392,692 rows (149,217 removed)

## 3. RFM Feature Engineering
Reference date: **2012-01-01** (one day after last transaction in dataset)

| Feature | Formula | Meaning |
|---|---|---|
| Recency | `ref_date - max(InvoiceDate)` per customer | Days since last purchase |
| Frequency | `COUNT(DISTINCT InvoiceNo)` per customer | Number of distinct purchase events |
| Monetary | `SUM(Quantity * UnitPrice)` per customer | Total customer spend |

**Result:** 4,338 unique customer RFM profiles

## 4. Transform & Scale
- **log1p transform** applied to Recency, Frequency, and Monetary to reduce
  right-skew (a small number of high-frequency/high-spend customers would
  otherwise dominate the distribution)
- **RobustScaler** applied after transform — chosen over StandardScaler
  because it scales using median/IQR, making it less sensitive to the
  legitimate outliers present in customer spend data

## 5. Output Artifacts
All outputs are DVC-tracked and must not be committed directly to Git.

| Artifact | Path | Purpose |
|---|---|---|
| Feature store | `data/processed/rfm_processed.csv` | Scaled RFM profiles for model training |
| Fitted scaler | `models/scaler.pkl` | Must be loaded by the Phase 5 prediction API to transform live incoming RFM values to match the model's training space |

**Important:** The scaler is fitted on training data and serialised via
pickle. The Phase 5 `/predict` API must load this exact scaler before
running inference. Without it, live predictions will be in a different
scale than the model expects and will be completely wrong.

## 6. Execution & Environment Dependencies
Pipeline is triggered via `dvc repro` only — scripts are never run manually.

### Required before running `dvc repro`:

**Step 1 — Create a `.env` file** in the project root with MLflow credentials:
MLFLOW_TRACKING_USERNAME=your_dagshub_username

MLFLOW_TRACKING_PASSWORD=your_dagshub_token

Without this, `train.py` will crash with a 403 authentication error when
attempting to connect to the DagsHub MLflow server.

**Step 2 — Configure local DVC remote credentials:**
```bash
dvc remote modify --local ovgu-remote user "your_ovgu_username"
dvc remote modify --local ovgu-remote password "your_nextcloud_app_password"
```

Without this, `dvc pull` and `dvc push` will fail to access remote storage.

These credentials are intentionally not tracked in Git for security reasons.

## 7. API Data Contract
The following schema defines the exact output of the data preparation
pipeline. Jash must use these types and boundaries for Pydantic validation
in the `/predict` endpoint.

| Field | Type | Description | Expected Range (post-scaling) |
|---|---|---|---|
| `CustomerID` | `str` | Unique customer identifier | Any string |
| `Recency` | `float` | log1p + RobustScaler transformed recency | Typically -2.0 to 5.0 |
| `Frequency` | `float` | log1p + RobustScaler transformed frequency | Typically -2.0 to 5.0 |
| `Monetary` | `float` | log1p + RobustScaler transformed monetary | Typically -2.0 to 5.0 |

**Note:** The API receives raw R, F, M values from the caller, applies
`log1p` then loads `models/scaler.pkl` to transform them before passing
to the model. The ranges above are post-transformation reference values
from the training data.

## 8. Verification
- MLflow connectivity to the shared DagsHub tracking server confirmed via
  a dummy run in `train.py`, validating the pipeline is ready for real
  modelling logic in Phase 3

## Next Steps
- Implement `src/train.py` — K-Means (k=2–10) with MLflow experiment logging,
  alternative model comparison (DBSCAN/GMM), model registration to Staging
- Implement `src/evaluate.py` — cluster metrics, profiling, business labelling