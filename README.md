# Used Car Price Prediction with Explainable AI

A machine learning project for predicting used car prices from Craigslist listings, with a strong emphasis on **Explainable AI (XAI)** techniques to interpret model decisions for non-technical stakeholders.

## Project Overview

**Goal:** Build interpretable ML models that predict used car prices and provide actionable insights — e.g., which features drive price up or down, and by how much.

**Dataset:** [Craigslist Cars and Trucks Data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) — ~426K listings scraped from Craigslist across the US.

**Target variable:** `price` (USD)

---

## Repository Structure

```
.
├── data/
│   ├── raw/            # Original dataset (not tracked in git)
│   └── processed/      # Cleaned & feature-engineered data
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb        # Data cleaning & feature engineering
│   ├── 03_Modeling.ipynb             # Model training & evaluation
│   └── 04_XAI.ipynb                  # SHAP, LIME, PDP, and more
├── src/
│   ├── data_processing.py            # Reusable preprocessing functions
│   ├── models.py                     # Model training utilities
│   └── xai_utils.py                  # XAI helper functions
├── reports/
│   └── figures/                      # Saved plots & charts
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_EDA.ipynb` | Data quality, distributions, correlations, geographic patterns |
| `02_Preprocessing.ipynb` | Outlier removal, encoding, feature engineering, train/test split |
| `03_Modeling.ipynb` | Linear Regression, Random Forest, XGBoost, LightGBM benchmarking |
| `04_XAI.ipynb` | SHAP values, LIME, PDP/ICE plots, permutation importance |

---

## XAI Methods Applied

- **SHAP** (SHapley Additive exPlanations) — global & local feature importance
- **LIME** (Local Interpretable Model-agnostic Explanations) — per-prediction explanations
- **Partial Dependence Plots (PDP)** — marginal effect of features on price
- **Individual Conditional Expectation (ICE)** — heterogeneous effects per instance
- **Permutation Feature Importance** — model-agnostic global ranking

---

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd used-car-price-prediction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place the dataset
cp vehicles.csv data/raw/vehicles.csv

# Launch Jupyter
jupyter notebook notebooks/
```

---

## Key Findings (Summary)

> Populated after modeling is complete.

---

## Team

- Member 1
- Member 2

University of Warsaw — Explainable AI Course, Spring 2026
