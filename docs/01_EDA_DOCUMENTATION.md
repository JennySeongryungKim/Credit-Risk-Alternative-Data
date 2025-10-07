# 01_eda.py — Exploratory Data Analysis Documentation

## Overview

This document summarizes the EDA outcomes produced by `01_EDA.py`, aligned with the pipeline’s stepwise structure (A–H). Results and file artifacts reflect your provided run outputs. Transformations (Yeo–Johnson windowizing), categorical encoding strategy, and SMOTE guidelines match the code in `01_EDA.py`.

---

### STEP A. Dataset Overview
- **Shape**: (307511, 336)
- **Average missing ratio**: 0.345
- **TARGET distribution**: {0: 282686, 1: 24825}

Artifacts:
- Target distribution plot: `eda_output/target_distribution.png`
- Missingness bar plot (top features): `eda_output/missing_top.png`

---

### STEP B. Numeric Features: Distributions, Skewness, Correlation
- **Top |skew|**: {"FLAG_MOBIL": 554.5367435977538, "BUREAU_AMT_CREDIT_MAX_OVERDUE_min": 444.59610411462734, "PREV_SELLERPLACE_AREA_min": 411.08801561455346, "INST_PAYMENT_RATIO_min": 394.48647187930544, "FLAG_DOCUMENT_12": 392.11477910204303, "AMT_INCOME_TOTAL": 391.5596541041876}
- **VIF (multicollinearity check)**: saved to `eda_output/vif.csv`

Artifacts:
- Correlation heatmap (top by |corr with TARGET|): `eda_output/corr_heatmap.png`
- KDE/Hist per top-skewed numeric features: `eda_output/kde_*.png` / `eda_output/hist_*.png`

---

### STEP C. Categorical Features: Distributions & Relationship to Target
- **Cramér’s V table**: saved to `eda_output/cramers_v.csv`

Artifacts:
- Top-K frequency with bad-rate overlays per categorical feature: `eda_output/cat_*.png`

---

### STEP D. Windowizing (Yeo–Johnson) Before/After Comparison
- Plots show reduced skewness and stabilized distributions after Yeo–Johnson.

Artifacts:
- Before/After histograms for selected skewed features: `eda_output/windowizing_*.png`

---

### STEP E. Categorical Encoding Strategy Justification
- **Cardinality/coverage report**: `eda_output/categorical_cardinality.csv`
- **Rule**:
  - If `nunique ≤ 2` → LabelEncoder (binary)
  - Else → Top-5 one-hot encoding to reduce sparsity and memory
- Higher Top-K coverage → lower information loss from truncation.

Notes:
- Mirrors production preprocessing to avoid train/serving skew.

---

### STEP F. Feature Rationale Summary
- **Feature explanations**: saved to `eda_output/feature_explanations.csv` (total documented: 334)

Notes:
- Rationale uses rule-based mapping consistent with naming patterns (e.g., `BUREAU_`, `PREV_`, `INST_`, ratios, temporal features).

---

### Encoding (for downstream analysis)
- Numeric features optionally transformed via Yeo–Johnson when skewed.
- Categorical features encoded per the rule above (binary label encoding or Top-5 one-hot).
- Outputs are consistent with `DataPreprocessor.encode_categorical` to match downstream training.

---

### STEP G. Mutual Information & RandomForest Importance
- **Mutual Information (top-100)**: saved to `eda_output/mi_top100.csv`
- **RandomForest AUC (validation)**: 0.7396

Artifacts:
- RandomForest top-30 feature importance plot: `eda_output/rf_importance_top30.png`
- RandomForest top-100 importance table: `eda_output/rf_importance_top100.csv`

---

### STEP H. Class Imbalance & SMOTE (Visualization Demo)
- **Before SMOTE class counts**: {0: 226148, 1: 19860}
- **After SMOTE class counts**: {0: 226148, 1: 113074}
- Note: Apply SMOTE only on the training fold to prevent leakage.

Artifacts:
- PCA (train) before SMOTE: `eda_output/pca_before_smote.png`
- PCA (train) after SMOTE: `eda_output/pca_after_smote.png`

---

## Reproducibility & Alignment with Code
- Steps and artifacts correspond to methods in `EDAReport` within `01_EDA.py` (A–H).
- Encoding and windowizing behavior aligns with `DataPreprocessor` to ensure consistent train-time and inference-time transformations.
- All saved CSV and PNG outputs are located under `artifact/EDA_output/`.
