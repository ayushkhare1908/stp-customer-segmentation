# STP Customer Segmentation (MLOps)

An end-to-end machine learning pipeline for e-commerce customer segmentation. This project operationalizes the clustering of retail customers based on purchasing behavior (RFM metrics) to drive personalized marketing strategies. 

It implements a strict MLOps process model, ensuring reproducibility, scalability, and continuous monitoring.

## System Architecture

* **Data Versioning:** DVC (Remote: OVGU Nextcloud WebDAV)
* **Experiment Tracking & Model Registry:** MLflow 
* **Containerization:** Docker (Single modular image)
* **Application / Utilization:** Streamlit (SPA with Marketing & MLOps views)

## Repository Structure

├── .dvc/                   # DVC configuration
├── .github/workflows/      # CI/CD Pipelines
├── data/                   # Tracked by DVC (Do not commit to Git)
│   ├── raw/                # Online Retail dataset
│   └── processed/          # Engineered RFM features
├── docs/                   # Scientific documentation and architecture
├── models/                 # Local model artifacts (Logged to MLflow)
├── src/                    # Production Code Base
│   ├── data_prep.py        # Data cleaning and feature engineering
│   ├── train.py            # Model selection and training
│   ├── evaluate.py         # Performance and business evaluation
│   └── app.py              # Single-page UI (Utilization phase)
└── Dockerfile              # Deployment environment

## Local Setup Instructions

**Prerequisites:** You must have Docker installed and an OVGU App Password generated for Nextcloud.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)[YOUR-USERNAME]/stp-customer-segmentation.git
   cd stp-customer-segmentation