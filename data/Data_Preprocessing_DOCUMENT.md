# Data Preprocessing Module Documentation

## Overview
This module provides comprehensive data preprocessing functionality for machine learning projects, specifically designed for financial/credit risk assessment datasets. It includes advanced features like windowizing (power transformation), feature engineering, categorical encoding, and data balancing using SMOTE.

**Due to the data size (approximately 2.5GB), only 1% of the entire file is provided as the result (preprocessed_data_sample_1pct.zip)

## File Information
- **File**: `data_preprocessing.py`
- **Purpose**: Data preprocessing pipeline for machine learning models
- **Target Domain**: Financial/Credit Risk Assessment
- **Key Features**: Windowizing, Feature Engineering, SMOTE Balancing, Multi-dataset Merging

## Dependencies
```python
# Core Data Processing
import pandas as pd
import numpy as np
import warnings
import pickle
import gc
import os
from datetime import datetime

# Data Preprocessing (SMOTE, windowizing)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

```

## Main Class: DataPreprocessor

### Class Overview
The `DataPreprocessor` class is the core component that handles all data preprocessing tasks including data loading, merging, feature engineering, windowizing, and preparation for machine learning models.

### Constructor
```python
def __init__(self):
    self.label_encoders = {}           # Stores LabelEncoder objects for categorical features
    self.scaler = StandardScaler()     # StandardScaler for feature normalization
    self.power_transformers = {}      # Stores PowerTransformer objects for windowizing
    self.alternative_features = []     # List of alternative feature names
    self.traditional_features = []     # List of traditional feature names
    self.all_features = []            # List of all feature names
```

### Methods

#### 1. `load_and_merge_data(self, data_paths)`
**Purpose**: Loads and merges multiple datasets into a single comprehensive dataset.

**Parameters**:
- `data_paths` (dict): Dictionary containing file paths for different datasets

**Expected data_paths structure**:
```python
data_paths = {
    'application': 'path/to/application_data.csv',
    'bureau': 'path/to/bureau_data.csv',                    # Optional
    'bureau_balance': 'path/to/bureau_balance_data.csv',    # Optional
    'previous_application': 'path/to/previous_app.csv',    # Optional
    'credit_card_balance': 'path/to/cc_balance.csv',       # Optional
    'pos_cash_balance': 'path/to/pos_balance.csv',         # Optional
    'installments_payments': 'path/to/installments.csv'    # Optional
}
```

**Process Flow**:
1. Loads main application data
2. Extracts target variable if present
3. Applies basic feature engineering
4. Merges auxiliary datasets (bureau, previous applications, etc.)
5. Aggregates auxiliary data to application level

**Returns**:
- `app_data` (DataFrame): Merged and engineered dataset
- `target` (array): Target variable array (if present)

#### 2. `engineer_basic_features(self, df)`
**Purpose**: Creates basic engineered features from the raw dataset.

**Features Created**:

- **Credit Ratios**:
  - `CREDIT_INCOME_RATIO = AMT_CREDIT / (AMT_INCOME_TOTAL + 1)`: Credit amount relative to income
  - `ANNUITY_INCOME_RATIO = AMT_ANNUITY / (AMT_INCOME_TOTAL + 1)`: Annuity payment relative to income
  - `CREDIT_GOODS_RATIO = AMT_CREDIT / (AMT_GOODS_PRICE + 1)`: Credit amount relative to goods price

- **Age and Employment**:
  - `AGE_YEARS = -DAYS_BIRTH / 365.25`: Age in years (converted from negative days)
  - `EMPLOYMENT_YEARS = clip(-DAYS_EMPLOYED / 365.25, 0, ∞)`: Employment duration in years (clipped to non-negative)
  - `DAYS_EMPLOYED_PERCENT = DAYS_EMPLOYED / (DAYS_BIRTH + 1)`: Employment days as percentage of age

- **External Source Aggregations** (for available EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 columns):
  - `EXT_SOURCE_MEAN`: Mean of external source scores
  - `EXT_SOURCE_STD`: Standard deviation of external source scores
  - `EXT_SOURCE_MIN`: Minimum external source score
  - `EXT_SOURCE_MAX`: Maximum external source score

**Returns**: DataFrame with additional engineered features

#### 3. `apply_windowizing(self, df, threshold=0.5)`
**Purpose**: Applies power transformation (Yeo-Johnson) to highly skewed numerical features.

**Parameters**:
- `df` (DataFrame): Input dataset
- `threshold` (float): Skewness threshold for transformation (default: 0.5)

**Target Features** (Windowizing 대상):
- **All AMT_* columns**: `AMT_CREDIT`, `AMT_INCOME_TOTAL`, `AMT_GOODS_PRICE`, `AMT_ANNUITY` and engineered ratios
- **Continuous variables**: `DAYS_EMPLOYED` and engineered ratios
- **Engineered ratios**: `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_GOODS_RATIO`

**Transformation Criterion**:
- **Condition**: `|skewness| > 0.5` → Apply PowerTransformer
- **Method**: `PowerTransformer(method='yeo-johnson', standardize=False)`
- **Storage**: Fitted transformers stored in `self.power_transformers[col]` for test-time reuse

**Process**:
1. Calculates skewness for each target feature
2. Applies Yeo-Johnson transformation if |skewness| > threshold
3. Stores transformers for later use on test data

**Returns**: DataFrame with transformed features


#### 4. `encode_categorical(self, df)`
**Purpose**: Encodes categorical variables using appropriate encoding strategies.

**Encoding Strategy**:

- **Binary Variables** (≤2 unique values): 
  - **Method**: `LabelEncoder` 
  - **Storage**: Stored in `self.label_encoders[col]`

- **Multi-category Variables** (>2 unique values): 
  - **Method**: One-hot encode top-5 frequent categories
  - **Action**: Drop original column after encoding

**Process**:
1. Identifies categorical columns (object dtype)
2. Skips ID columns (`SK_ID_CURR`, `SK_ID_PREV`)
3. Applies appropriate encoding based on cardinality
4. Stores encoders for later use

**Returns**: DataFrame with encoded categorical features

#### 5. `separate_features(self, df)`
**Purpose**: Separates features into alternative and traditional categories.

**Feature Classification Rules**:

- **Alternative Features** (대안적 특성):
  - **Keywords**: `FLAG_`, `EXT_SOURCE`, `REGION_`, `OBS_`, `DEF_`, `EMAIL`, `PHONE`, `MOBIL`, `SOCIAL`
  - **Characteristics**: Behavioral, demographic, and external indicator features

- **Traditional Features** (전통적 특성):
  - **Keywords**: `AMT_`, `DAYS_`, `CNT_`, `CREDIT`, `INCOME`, `BUREAU_`, `PREV_`, `ANNUITY`
  - **Characteristics**: Financial amounts, time-based features, and credit history features

**Returns**: Tuple of (alternative_features, traditional_features)

#### 6. `preprocess_and_save(self, train_paths, test_paths=None)`
**Purpose**: Main preprocessing pipeline that orchestrates all preprocessing steps.

**Parameters**:
- `train_paths` (dict): Training data file paths
- `test_paths` (dict, optional): Test data file paths

**Process Flow**:
1. **Data Loading**: Loads and merges training datasets
2. **Data Cleaning**: Handles infinite values and missing data
3. **Windowizing**: Applies power transformation to skewed features
4. **Categorical Encoding**: Encodes categorical variables
5. **Feature Separation**: Categorizes features into alternative/traditional
6. **Train-Validation Split**: Creates 80/20 split with stratification
7. **SMOTE Balancing**: Applies SMOTE with 50% sampling ratio
8. **Scaling**: Applies StandardScaler to all feature sets
9. **Data Saving**: Saves preprocessed data and preprocessor objects

**Output Datasets**:
- `all`: All features combined
- `traditional`: Only traditional financial features
- `alternative`: Only alternative behavioral features

**Saved Files**:
- `preprocessed_data.pkl`: Preprocessed datasets
- `preprocessor.pkl`: Preprocessor object for test data

**Returns**: Dictionary containing preprocessed datasets

## Key Features

### 1. Windowizing (Power Transformation)
- Uses Yeo-Johnson transformation for skewed features
- Automatically detects highly skewed features (|skewness| > 0.5)
- Stores transformers for consistent application to test data

### 2. Multi-Dataset Integration
- Supports merging of multiple auxiliary datasets
- Aggregates auxiliary data to application level
- Handles missing datasets gracefully

### 3. Feature Engineering
- Creates meaningful financial ratios
- Handles age and employment calculations
- Aggregates external source scores

### 4. Categorical Encoding
- Intelligent encoding based on cardinality
- Preserves information while reducing dimensionality
- Handles missing values appropriately

### 5. Data Balancing
- Uses SMOTE for handling class imbalance
- Configurable sampling ratio (default: 50%)
- Maintains data quality while balancing classes

### 6. Feature Categorization
- Separates features into alternative and traditional categories
- Enables separate modeling approaches
- Supports ensemble methods

## Usage Example

```python
# Initialize preprocessor
preprocessor = DataPreprocessor()

# Define data paths
train_paths = {
    'application': 'train/application_train.csv',
    'bureau': 'train/bureau.csv',
    'bureau_balance': 'train/bureau_balance.csv',
    'previous_application': 'train/previous_application.csv',
    'credit_card_balance': 'train/credit_card_balance.csv',
    'pos_cash_balance': 'train/POS_CASH_balance.csv',
    'installments_payments': 'train/installments_payments.csv'
}

# Run preprocessing
datasets = preprocessor.preprocess_and_save(train_paths)

# Access preprocessed data
all_features_data = datasets['all']
traditional_features_data = datasets['traditional']
alternative_features_data = datasets['alternative']
```

## Output Structure

Each dataset in the output dictionary contains:
```python
{
    'X_train': numpy.ndarray,    # Training features (scaled and balanced)
    'X_val': numpy.ndarray,      # Validation features (scaled)
    'y_train': numpy.ndarray,    # Training targets (balanced)
    'y_val': numpy.ndarray,      # Validation targets
    'features': list,            # Feature names
    'scaler': StandardScaler     # Fitted scaler object
}
```

## Memory Management
- Uses garbage collection (`gc.collect()`) to free memory
- Processes data in chunks to handle large datasets
- Saves intermediate results to disk

## Error Handling
- Gracefully handles missing datasets
- Continues processing even if individual transformations fail
- Provides detailed logging of processing steps

## Performance Considerations
- Uses efficient pandas operations for data manipulation
- Implements memory-efficient processing for large datasets
- Supports parallel processing where possible
- Optimized for financial datasets with many categorical variables


## Notes
- This module is specifically designed for financial/credit risk datasets
- The windowizing technique helps normalize highly skewed financial features
- SMOTE balancing helps address class imbalance common in credit risk datasets
- Feature separation enables different modeling approaches for different feature types
- All transformations are stored for consistent application to test data
