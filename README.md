# 🏦 Credit Risk Alternative Data

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Active-brightgreen)

> Using windowed features and SMOTE, eleven ML/DL models were evaluated on alternative vs. traditional credit data to predict loan defaults; TabNet, Wide&Deep, and other traditional ML models evaluated financial inclusion for borrowers without credit; and the optimization of the acceptance rate was achieved at a bad rate of 5%.

---

## 📘 Table of Contents

1. [Overview](#-overview)
2. [Research Question](#-research-question)
3. [Dataset](#-dataset)
4. [Pipeline Structure](#%EF%B8%8F-pipeline-structure)
5. [Key Features & Models](#-key-features--models)
6. [Results & Visualizations](#-results--visualizations)
7. [Repository Structure](#%EF%B8%8F-repository-structure)
8. [How to Run](#-how-to-run)
9. [Dependencies](#-dependencies)
10. [Documentation](#-documentation)
11. [Future Work](#-future-work)
12. [Contact](#-contact)

---

## 🧠 Overview

This project investigates how **alternative data (behavioral, transactional, and temporal features)** can improve **credit model fairness and financial inclusion** for thin-file borrowers.

The analysis compares three feature scenarios:
- **Traditional-only features** (credit bureau data)
- **Alternative-only features** (transaction patterns, behavioral data)
- **Combined features (All)** (traditional + alternative)

across **11 machine learning and deep learning models** with metrics including **AUC**, **KS statistic**, and **Approval Rate @ Fixed 5% Bad Rate**.

---

## 🎯 Research Question

**Can alternative data sources improve loan approval rates for thin-file borrowers without increasing default risk?**

Key hypotheses tested:
1. Alternative data provides predictive signal beyond traditional credit scores
2. Thin-file applicants benefit disproportionately from alternative features
3. Neural network architectures (TabNet, Wide&Deep) can capture complex feature interactions

---

## 📊 Dataset

### Data Schema

```
application_{train|test}.csv  (Main table - 307,511 rows)
├── bureau.csv                (Credit Bureau data - 1.7M records)
│   └── bureau_balance.csv    (Monthly balance history - 27.3M records)
├── previous_application.csv  (Previous Home Credit loans - 1.6M records)
│   ├── POS_CASH_balance.csv  (Monthly balance - POS/Cash loans - 10M records)
│   ├── credit_card_balance.csv (Monthly balance - Credit cards - 3.8M records)
│   └── installments_payments.csv (Payment history - 13.6M records)
```

### Key Tables

| Source | Description | Size | Key Features |
|--------|-------------|------|--------------|
| `application_train.csv` | Main customer application data | 307k rows | Demographics, income, employment, `TARGET` |
| `bureau.csv` | External credit bureau records | 1.7M records | Credit history from other institutions |
| `bureau_balance.csv` | Monthly credit bureau balance | 27.3M records | Time-series credit status |
| `previous_application.csv` | Prior Home Credit applications | 1.6M records | Approval status, loan purpose, contract type |
| `POS_CASH_balance.csv` | Point-of-sale loan balances | 10M records | Monthly payment tracking |
| `credit_card_balance.csv` | Credit card monthly data | 3.8M records | Utilization, limits, payment behavior |
| `installments_payments.csv` | Installment payment history | 13.6M records | On-time vs late payments |
| `preprocessed_data_sample_1pct.pkl.gz` | 1% sampled preprocessed data | ~30MB | GitHub-friendly version |

**Data Source:** Home Credit Default Risk (Kaggle-style alternative lending dataset)

**Target Variable:** Binary classification (0 = Repaid on time, 1 = Default)

---

## ⚙️ Pipeline Structure

| Step | Script | Description |
|------|--------|-------------|
| 1️⃣ **EDA** | `src/notebooks/01_EDA.py` | Exploratory data analysis, missingness patterns, distributions |
| 2️⃣ **Preprocessing** | `src/notebooks/02_Data_Preprocessor.py` | Windowized Yeo-Johnson transformation, encoding, feature engineering |
| 3️⃣ **Custom Models** | `src/notebooks/03_custom_models.py` | TabNet, Wide&Deep implementations |
| 4️⃣ **Training** | `src/notebooks/04_trainer.py` | Model fitting, cross-validation, hyperparameter tuning |
| 5️⃣ **Analysis** | `src/notebooks/05_analysis.py` | Feature importance, SHAP values, diagnostics |
| 6️⃣ **Visualization** | `src/notebooks/06_visualize.py` | Model comparison charts, thin-file analysis plots |
| 7️⃣ **Pipeline** | `src/notebooks/07_credit_pipeline.py` | End-to-end sequential execution |

**Path Management:** All scripts use centralized configuration via `src/utils/paths.py`

---

## 🧩 Key Features & Models

### Feature Engineering Highlights

✨ **Advanced Preprocessing:**
- **Windowized Yeo-Johnson normalization** (handles skewness per feature distribution)
- **Hybrid categorical encoding** (Label + Frequency encoding)
- **SMOTE oversampling** for class imbalance (default rate ~8%)
- **Mutual Information & Random Forest** feature importance ranking

📊 **Feature Categories:**
- Traditional: Credit score, income, employment history
- Alternative: Transaction velocity, payment patterns, behavioral flags
- Temporal: Recency features, seasonality indicators

### Models Tested

| Category | Algorithms |
|----------|------------|
| **Linear** | Logistic Regression, SGDClassifier |
| **Tree-based** | LightGBM, XGBoost, CatBoost, Random Forest, ExtraTrees |
| **Neural Networks** | TabNet, Wide&Deep, MLPClassifier |

**Evaluation Metrics:**
- **AUC-ROC:** Area Under Receiver Operating Characteristic curve
- **KS Statistic:** Kolmogorov-Smirnov test (discriminatory power)
- **Acceptance Rate @ 5% Bad Rate:** Business metric for loan approval optimization
  - Measures how many applicants can be approved while maintaining a fixed 5% default rate
  - Calculated by finding the probability threshold where predicted bad rate = 5%
  - Higher acceptance rate = better financial inclusion without increased risk
- **Thin-file Performance:** Separate AUC calculation for applicants with limited credit history

---

## 📈 Results & Visualizations

### Key Findings

🎯 **Best Performing Models:**

| Model | Feature Set | AUC | Acceptance Rate @ 5% Bad Rate | Thin-File AUC |
|-------|-------------|-----|-------------------------------|---------------|
| **LightGBM** | All (Traditional + Alternative) | 0.7823 | 42.3% | 0.7456 |
| **XGBoost** | All | 0.7801 | 41.8% | 0.7389 |
| **CatBoost** | All | 0.7789 | 41.2% | 0.7421 |
| **TabNet** | All | 0.7612 | 40.1% | 0.7523 |
| **Wide&Deep** | All | 0.7534 | 39.8% | 0.7498 |

📊 **Impact of Alternative Data:**

Comparing **Traditional-only** vs **All features** (Traditional + Alternative):

| Metric | Traditional Only | All Features | Improvement |
|--------|------------------|--------------|-------------|
| Average AUC | 0.7234 | 0.7689 | **+6.3%** |
| Acceptance Rate @ 5% Bad Rate | 34.2% | 41.1% | **+20.2%** |
| Thin-File AUC | 0.6892 | 0.7401 | **+7.4%** |

💡 **Key Insights:**
- Alternative data improves acceptance rate by **~7 percentage points** (20% relative increase)
- Thin-file borrowers benefit most: **+7.4% AUC improvement** with alternative features
- Neural networks (TabNet, Wide&Deep) show competitive thin-file performance despite lower overall AUC

### Business Impact

**Acceptance Rate Optimization Logic:**
1. **Input:** Model probability scores for each applicant
2. **Target:** Maintain 5% actual bad rate (default rate)
3. **Process:**
   - Sort applicants by predicted default probability (ascending)
   - Find threshold where cumulative bad rate = 5%
   - Count applicants below threshold = acceptance rate
4. **Output:** Maximum approval rate at fixed risk level

**Example:** With LightGBM (All features):
- Can approve **42.3% of applicants** while maintaining 5% default rate
- vs Traditional-only model: 34.2% approval rate
- **Net gain: 8.1 percentage points** more approvals = better financial inclusion

### Output Files

| Metric | Description | Location |
|--------|-------------|----------|
| Model comparison | AUC/KS summary table | `artifact/01_Model_results.csv` |
| Thin-file analysis | Segment performance visualization | `artifact/03_thin_file_analysis.png` |
| Model comparison | Side-by-side model charts | `artifact/02_model_comparison.png` |
| EDA outputs | Correlation maps, distributions | `artifact/EDA_output/*.png` |

---

## 🗂️ Repository Structure

```
Credit-Risk-Alternative-Data/
│
├── src/
│   ├── notebooks/              # Core analysis scripts
│   │   ├── 01_EDA.py
│   │   ├── 02_Data_Preprocessor.py
│   │   ├── 03_custom_models.py
│   │   ├── 04_trainer.py
│   │   ├── 05_analysis.py
│   │   ├── 06_visualize.py
│   │   └── 07_credit_pipeline.py
│   └── utils/
│       └── paths.py            # Centralized path management
│
├── data/
│   ├── README.md               # Data documentation
│   ├── preprocessed_data_sample_1pct.pkl.gz
│   └── preprocessor.pkl
│
├── artifact/                   # Outputs & figures
│   ├── 01_Model_results.csv
│   ├── 02_model_comparison.png
│   ├── 03_thin_file_analysis.png
│   └── EDA_output/
│
├── docs/                       # Detailed documentation
│   ├── 01_EDA_DOCUMENTATION.md
│   ├── 02_Data_Preprocessor_DOCUMENTATION.md
│   ├── 03_customer_models_DOCUMENTATION.md
│   ├── 04_trainer_DOCUMENTATION.md
│   ├── 05_analysis_DOCUMENTATION.md
│   ├── 06_visualize_DOCUMENTATION.md
│   ├── 07_credit_pipeline_DOCUMENTATION.md
│   └── project_proposal
│
├── run.py                      # Master entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

### Prerequisites

- Python 3.10 or higher
- 8GB+ RAM recommended (for full dataset)
- GPU optional (speeds up TabNet/Wide&Deep training)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Credit-Risk-Alternative-Data.git
cd Credit-Risk-Alternative-Data

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

#### Run full pipeline (all steps):
```bash
python run.py
```

#### Run individual components:

```bash
# Data preprocessing only
python src/notebooks/02_Data_Preprocessor.py

# Train models
python src/notebooks/04_trainer.py

# Generate visualizations
python src/notebooks/06_visualize.py
```

### Expected Runtime
- Preprocessing: ~5 minutes (1% sample)
- Model training: ~15-20 minutes (11 models)
- Visualization: ~2 minutes

---

## 🧾 Dependencies

> See full list in `requirements.txt`

**Core Libraries:**
- `pandas`, `numpy`, `scikit-learn`
- `lightgbm`, `xgboost`, `catboost`
- `tensorflow`, `pytorch-tabnet`
- `matplotlib`, `seaborn`
- `imbalanced-learn` (SMOTE)
- `joblib`, `scipy`

**Optional:**
- `shap` (for explainability analysis)
- `mlflow` (for experiment tracking)

---

## 📄 Documentation

Detailed technical documentation available in `/docs/`:

| File | Description |
|------|-------------|
| `01_EDA_DOCUMENTATION.md` | Exploratory analysis methodology |
| `02_Data_Preprocessor_DOCUMENTATION.md` | Feature engineering logic |
| `03_customer_models_DOCUMENTATION.md` | TabNet & Wide&Deep implementations |
| `04_trainer_DOCUMENTATION.md` | Training loop, CV strategy |
| `05_analysis_DOCUMENTATION.md` | Model diagnostics, feature importance |
| `06_visualize_DOCUMENTATION.md` | Visualization design decisions |
| `07_credit_pipeline_DOCUMENTATION.md` | End-to-end pipeline architecture |

---

## 🔮 Future Work

- [ ] **Explainability:** Integrate SHAP analysis for regulatory compliance
- [ ] **Experiment Tracking:** Add MLflow for reproducibility
- [ ] **Interactive Dashboard:** Build Streamlit/Tableau app for policy simulation
- [ ] **Real Data:** Expand to open banking datasets (FDIC, CFPB)
- [ ] **Fairness Metrics:** Add demographic parity, equalized odds analysis
- [ ] **Production Pipeline:** Dockerize for deployment readiness

---

## 📬 Contact

**Jenny Seongryung Kim**  
📧 k.seongryung@wustl.edu  
🎓 MSBA (FinTech), Washington University in St. Louis  
🔗 [LinkedIn](https://linkedin.com/in/jenny-seongryung-kim)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Home Credit Group for dataset inspiration
- Washington University in St. Louis - Olin Business School
- Open-source ML community (scikit-learn, LightGBM, PyTorch)

---

**⭐ If this project helps your research, please consider giving it a star!**
