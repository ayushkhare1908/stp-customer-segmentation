# Data Preparation Log — `src/data_prep.py`

**Author:** Ayush | **Phase:** 2 — Data & EDA | **Date:** [add today's date]

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

## 5. Output
- File: `data/processed/rfm_processed.csv`
- Columns: `CustomerID`, `Recency`, `Frequency`, `Monetary` (scaled)
- Tracked via DVC, not committed directly to Git

## 6. Execution
- Pipeline triggered via `dvc repro` only (per `dvc.yaml` stage `data_preparation`)
- Scripts are never run manually — DVC manages dependency tracking and
  re-execution of only changed stages
- `dvc.lock` updated to reflect new pipeline state after each run

## 7. Verification
- MLflow connectivity to the shared DagsHub tracking server confirmed via a
  dummy run in `train.py`, validating the pipeline is ready for real
  modelling logic in Phase 3

## Next Steps
- Implement `src/train.py` — K-Means (k=2–10) with MLflow experiment logging,
  alternative model comparison (DBSCAN/GMM), model registration to Staging
- Implement `src/evaluate.py` — cluster metrics, profiling, business labelling