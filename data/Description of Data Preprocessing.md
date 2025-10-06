# Data Preprocessing (Windowizing + SMOTE) Pipeline

**TL;DR**

* **Feature engineering** → **Windowizing (Yeo-Johnson)** → **Encoding** → **Train/Val split** → **Scaling** → **SMOTE (0.5)** → **Save artifacts**
* Artifacts: `preprocessed_data.pkl` (X/y folds, scalers), `preprocessor.pkl` (encoders / transformers / feature lists)

---

## 1) What this does

* Merge multiple sources by **SK_ID_CURR** (`application`, `bureau`, `previous_application`, `credit_card_balance`, `pos_cash_balance`, `installments_payments`)
* Create **engineered features** (credit/income & annuity ratios, age/employment years, EXT_SOURCE stats)
* **Windowizing**: apply Yeo-Johnson to highly skewed numeric columns (`threshold=0.5`, no standardization inside)
* **Categoricals**: binary → `LabelEncoder`; otherwise Top-5 one-hot
* Split features into **alternative** vs **traditional** via keyword rules
* **Stratified** train/val = **80/20**, `random_state=42`
* **Standardize** per feature set (all / traditional / alternative)
* **SMOTE(0.5)** to mitigate imbalance with memory-friendly oversampling
* Save ready-to-train arrays + scalers + preprocessor object

---

## 2) Inputs

| Key in `data_paths`     | Expected file | Notes                                                                 |
| ----------------------- | ------------- | --------------------------------------------------------------------- |
| `application`           | CSV           | Required. Detects `TARGET` automatically if present                   |
| `bureau`                | CSV           | Optional. Aggregated and merged with `bureau_balance` if provided     |
| `bureau_balance`        | CSV           | Optional. Aggregations by `SK_ID_BUREAU`: min/max/mean/status-0 count |
| `previous_application`  | CSV           | Optional. Numeric mean/max/min by `SK_ID_CURR`                        |
| `credit_card_balance`   | CSV           | Optional. Numeric mean/max/min by `SK_ID_CURR`                        |
| `pos_cash_balance`      | CSV           | Optional. Numeric mean/max/min by `SK_ID_CURR`                        |
| `installments_payments` | CSV           | Optional. Creates `PAYMENT_DIFF`, `PAYMENT_RATIO`, then aggregates    |

---

## 3) Feature Engineering

* **Credit/Income ratio**: `CREDIT_INCOME_RATIO = AMT_CREDIT / (AMT_INCOME_TOTAL + 1)`
* **Annuity/Income ratio**: `ANNUITY_INCOME_RATIO = AMT_ANNUITY / (AMT_INCOME_TOTAL + 1)`
* **Credit/Goods ratio**: `CREDIT_GOODS_RATIO = AMT_CREDIT / (AMT_GOODS_PRICE + 1)`
* **Age/Employment**:
  `AGE_YEARS = -DAYS_BIRTH/365.25`
  `EMPLOYMENT_YEARS = clip(-DAYS_EMPLOYED/365.25, 0, ∞)`
* **External sources**: `EXT_SOURCE_{MEAN,STD,MIN,MAX}` (for available columns)

**Alternative vs Traditional keyword rules**

* **Alternative**: `FLAG_`, `EXT_SOURCE`, `REGION_`, `OBS_`, `DEF_`, `EMAIL`, `PHONE`, `MOBIL`, `SOCIAL`
* **Traditional**: `AMT_`, `DAYS_`, `CNT_`, `CREDIT`, `INCOME`, `BUREAU_`, `PREV_`, `ANNUITY`

---

## 4) Windowizing (Yeo-Johnson)

* **Targets**: all `AMT_*` + continuous variables like `AMT_CREDIT`, `AMT_INCOME_TOTAL`, `AMT_GOODS_PRICE`, `AMT_ANNUITY`, `DAYS_EMPLOYED`, ratios
* **Criterion**: `|skewness| > 0.5` → `PowerTransformer(method='yeo-johnson', standardize=False)`
* Fitted transformers stored in `self.power_transformers[col]` for test-time reuse

---

## 5) Encoding

* **Binary** (≤2 unique): `LabelEncoder` → stored in `self.label_encoders[col]`
* **Multi-class**: one-hot encode top-5 frequent categories, drop original column

---

## 6) Splitting, Scaling, SMOTE

* **Split**: `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`
* **Scaling**: `StandardScaler` trained per feature set (`all`, `traditional`, `alternative`)
* **SMOTE**: `sampling_strategy=0.5` (positive ≈ 0.5 × negative) for memory efficiency

---

## 7) Outputs (Artifacts)

* **`preprocessed_data.pkl`**

  * Contains:

    * `X_train`, `X_val` (scaled numpy arrays)
    * `y_train`, `y_val`
    * `features` (column names)
    * `scaler` (StandardScaler)
* **`preprocessor.pkl`**

  * Contains `DataPreprocessor` instance (encoders, transformers, feature type lists)

---

## 8) How to run

```python
from data_preprocessor import DataPreprocessor

train_paths = {
    "application": "data/application_train.csv",
    "bureau": "data/bureau.csv",
    "bureau_balance": "data/bureau_balance.csv",
    "previous_application": "data/previous_application.csv",
    "credit_card_balance": "data/credit_card_balance.csv",
    "pos_cash_balance": "data/pos_cash_balance.csv",
    "installments_payments": "data/installments_payments.csv",
}

pre = DataPreprocessor()
datasets = pre.preprocess_and_save(train_paths)
# → artifacts: preprocessed_data.pkl, preprocessor.pkl
```

---

## 9) Inference Tips

* Load `preprocessor.pkl` for consistent **windowizing + encoding + scaling**
* Missing columns in new data are skipped automatically
* For consistent shape alignment, reorder columns by the saved `features` list before prediction

---

## 10) Requirements

* `pandas>=2.0.0`, `numpy>=1.24.0`, `scikit-learn>=1.3.0`, `imbalanced-learn>=0.11.0`
* `lightgbm>=4.0.0`, `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `joblib>=1.3.0`
* *(Optional)* `scipy>=1.11.0`, `pandas[performance]>=2.0.0`

> Install via:
> `pip install -r requirements.txt`

---

## 11) Reproducibility & Notes

* Use `random_state=42`, `stratify=y` for consistent splits
* Replace infinities → NaN → 0:
  `df.replace([np.inf, -np.inf], np.nan).fillna(0)`
* For large data, consider dimensionality reduction (PCA, autoencoder) before SMOTE

