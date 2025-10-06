# -*- coding: utf-8 -*-
### 🧩 Component: `DataPreprocessor` (Windowizing + SMOTE)

**Purpose**
- Merge raw tables by `SK_ID_CURR`, engineer features, de-skew with **Yeo-Johnson**, encode categoricals, split/scale, and balance using **SMOTE(0.5)**.
- Persist artifacts for training/inference reuse.

**Inputs**
- `data/application_train.csv` (required, must contain `TARGET`)
- `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`,
  `credit_card_balance.csv`, `POS_CASH_balance.csv`, `installments_payments.csv`

**Outputs (under `artifact/`)**
- `preprocessed_data.pkl`: dict with keys `all`, `traditional`, `alternative`  
  → each contains `X_train`, `X_val`, `y_train`, `y_val`, `features`, `scaler`
- `preprocessor.pkl`: fitted `DataPreprocessor` (encoders, YJ transformers, feature lists)

**Run**
```python
from src.pipeline.data_preprocessor import DataPreprocessor
from src.utils.paths import data_path

train_paths = {
    "application": data_path("application_train.csv"),
    "bureau": data_path("bureau.csv"),
    "bureau_balance": data_path("bureau_balance.csv"),
    "previous_application": data_path("previous_application.csv"),
    "credit_card_balance": data_path("credit_card_balance.csv"),
    "pos_cash_balance": data_path("POS_CASH_balance.csv"),
    "installments_payments": data_path("installments_payments.csv"),
}
pre = DataPreprocessor()
datasets = pre.preprocess_and_save(train_paths)
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import gc
import os
from datetime import datetime
from typing import Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles all data preprocessing including windowizing"""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.power_transformers = {}  # Store transformers for test data
        self.alternative_features = []
        self.traditional_features = []
        self.all_features = []


    def load_and_merge_data(self, data_paths):
      """Load and merge all datasets"""
      print("Loading datasets...")

      # Load main application data
      app_data = pd.read_csv(data_paths['application'])
      print(f"  Application data shape: {app_data.shape}")

     # Store target if it exists
      if 'TARGET' in app_data.columns:
          target = app_data['TARGET'].values
      else:
          target = None

      # Basic feature engineering
      app_data = self.engineer_basic_features(app_data)

     # 1. Bureau + Bureau Balance
      if 'bureau' in data_paths and os.path.exists(data_paths['bureau']):
          print("  Loading bureau data...")
          bureau = pd.read_csv(data_paths['bureau'])

          # Merge bureau_balance if available
          if 'bureau_balance' in data_paths and os.path.exists(data_paths['bureau_balance']):
              print("  Loading bureau_balance data...")
              bureau_balance = pd.read_csv(data_paths['bureau_balance'])

              # Aggregate bureau_balance to bureau level
              bb_agg = bureau_balance.groupby('SK_ID_BUREAU').agg({
                  'MONTHS_BALANCE': ['min', 'max', 'mean'],
                  'STATUS': lambda x: (x == '0').sum()
              }).fillna(0)
              bb_agg.columns = ['BB_' + '_'.join(col).strip() for col in bb_agg.columns.values]
              bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')

          # Aggregate bureau to application level
          bureau_numeric = bureau.select_dtypes(include=[np.number])
          bureau_agg = bureau_numeric.groupby('SK_ID_CURR').agg({
              col: ['mean', 'max', 'min'] for col in bureau_numeric.columns
              if col not in ['SK_ID_CURR', 'SK_ID_BUREAU']
          }).fillna(0)
          bureau_agg.columns = ['BUREAU_' + '_'.join(col).strip() for col in bureau_agg.columns.values]
          app_data = app_data.merge(bureau_agg, on='SK_ID_CURR', how='left')

      # 2. Previous Application
      if 'previous_application' in data_paths and os.path.exists(data_paths['previous_application']):
          print("  Loading previous application data...")
          prev_app = pd.read_csv(data_paths['previous_application'])
          prev_numeric = prev_app.select_dtypes(include=[np.number])
          prev_agg = prev_numeric.groupby('SK_ID_CURR').agg({
              col: ['mean', 'max', 'min'] for col in prev_numeric.columns
              if col not in ['SK_ID_CURR', 'SK_ID_PREV']
          }).fillna(0)
          prev_agg.columns = ['PREV_' + '_'.join(col).strip() for col in prev_agg.columns.values]
          app_data = app_data.merge(prev_agg, on='SK_ID_CURR', how='left')

      # 3. Credit Card Balance
      if 'credit_card_balance' in data_paths and os.path.exists(data_paths['credit_card_balance']):
          print("  Loading credit card balance data...")
          cc_balance = pd.read_csv(data_paths['credit_card_balance'])
          cc_numeric = cc_balance.select_dtypes(include=[np.number])
          cc_agg = cc_numeric.groupby('SK_ID_CURR').agg({
              col: ['mean', 'max', 'min'] for col in cc_numeric.columns
              if col not in ['SK_ID_CURR', 'SK_ID_PREV']
          }).fillna(0)
          cc_agg.columns = ['CC_' + '_'.join(col).strip() for col in cc_agg.columns.values]
          app_data = app_data.merge(cc_agg, on='SK_ID_CURR', how='left')

      # 4. POS Cash Balance
      if 'pos_cash_balance' in data_paths and os.path.exists(data_paths['pos_cash_balance']):
          print("  Loading POS cash balance data...")
          pos_balance = pd.read_csv(data_paths['pos_cash_balance'])
          pos_numeric = pos_balance.select_dtypes(include=[np.number])
          pos_agg = pos_numeric.groupby('SK_ID_CURR').agg({
              col: ['mean', 'max', 'min'] for col in pos_numeric.columns
              if col not in ['SK_ID_CURR', 'SK_ID_PREV']
          }).fillna(0)
          pos_agg.columns = ['POS_' + '_'.join(col).strip() for col in pos_agg.columns.values]
          app_data = app_data.merge(pos_agg, on='SK_ID_CURR', how='left')

      # 5. Installments Payments
      if 'installments_payments' in data_paths and os.path.exists(data_paths['installments_payments']):
          print("  Loading installments payments data...")
          installments = pd.read_csv(data_paths['installments_payments'])

          # Calculate payment difference
          installments['PAYMENT_DIFF'] = installments['AMT_PAYMENT'] - installments['AMT_INSTALMENT']
          installments['PAYMENT_RATIO'] = installments['AMT_PAYMENT'] / (installments['AMT_INSTALMENT'] + 0.0001)

          inst_numeric = installments.select_dtypes(include=[np.number])
          inst_agg = inst_numeric.groupby('SK_ID_CURR').agg({
              col: ['mean', 'max', 'min'] for col in inst_numeric.columns
              if col not in ['SK_ID_CURR', 'SK_ID_PREV']
          }).fillna(0)
          inst_agg.columns = ['INST_' + '_'.join(col).strip() for col in inst_agg.columns.values]
          app_data = app_data.merge(inst_agg, on='SK_ID_CURR', how='left')

      print(f"  Final merged shape: {app_data.shape}")

      return app_data, target

    def engineer_basic_features(self, df):
        """Create basic engineered features"""
        # Credit ratios
        if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)

        if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)

        if 'AMT_CREDIT' in df.columns and 'AMT_GOODS_PRICE' in df.columns:
            df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)

        # Age and employment
        if 'DAYS_BIRTH' in df.columns:
            df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25

        if 'DAYS_EMPLOYED' in df.columns:
            df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365.25
            df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].clip(lower=0)

        if 'DAYS_EMPLOYED' in df.columns and 'DAYS_BIRTH' in df.columns:
            df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1)

        # External sources
        ext_source_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        ext_source_cols = [col for col in ext_source_cols if col in df.columns]
        if len(ext_source_cols) > 0:
            df['EXT_SOURCE_MEAN'] = df[ext_source_cols].mean(axis=1)
            df['EXT_SOURCE_STD'] = df[ext_source_cols].std(axis=1)
            df['EXT_SOURCE_MIN'] = df[ext_source_cols].min(axis=1)
            df['EXT_SOURCE_MAX'] = df[ext_source_cols].max(axis=1)

        return df

    def apply_windowizing(self, df, threshold=0.5):
        """
        Apply power transformation (Yeo-Johnson) to skewed numerical features

        Args:
            df: DataFrame
            threshold: Skewness threshold for transformation
        Returns:
            DataFrame with transformed features
        """
        print("  Applying windowizing (Yeo-Johnson transformation) to skewed features...")

        # Target columns for windowizing
        target_cols = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE',
                      'AMT_ANNUITY', 'DAYS_EMPLOYED', 'CREDIT_INCOME_RATIO',
                      'ANNUITY_INCOME_RATIO', 'CREDIT_GOODS_RATIO']

        # Also check all AMT_ columns
        amt_cols = [col for col in df.columns if 'AMT_' in col]
        target_cols.extend(amt_cols)
        target_cols = list(set(target_cols))  # Remove duplicates

        transformed_cols = []

        for col in target_cols:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                # Check skewness
                skewness = df[col].skew()

                # Transform if highly skewed
                if abs(skewness) > threshold:
                    try:
                        # Use Yeo-Johnson transformation
                        pt = PowerTransformer(method='yeo-johnson', standardize=False)
                        df[col] = pt.fit_transform(df[[col]].values.reshape(-1, 1)).flatten()
                        self.power_transformers[col] = pt  # Save for test data
                        transformed_cols.append(col)
                    except:
                        continue

        print(f"    Transformed {len(transformed_cols)} skewed features")
        if len(transformed_cols) > 0 and len(transformed_cols) <= 15:
            print(f"    Features: {', '.join(transformed_cols[:15])}")

        return df

    def encode_categorical(self, df):
        """Encode categorical variables"""
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        print(f"  Encoding {len(categorical_cols)} categorical columns...")

        for col in categorical_cols:
            if col in ['SK_ID_CURR', 'SK_ID_PREV']:
                continue

            if df[col].nunique() <= 2:
                # Binary encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].fillna('Missing'))
                self.label_encoders[col] = le
            else:
                # One-hot encoding for top 5 categories
                top_cats = df[col].value_counts().head(5).index.tolist()
                for cat in top_cats:
                    df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
                df = df.drop(col, axis=1)

        return df

    def separate_features(self, df):
        """Separate alternative and traditional features"""
        alternative_keywords = ['FLAG_', 'EXT_SOURCE', 'REGION_', 'OBS_', 'DEF_',
                              'EMAIL', 'PHONE', 'MOBIL', 'SOCIAL']
        traditional_keywords = ['AMT_', 'DAYS_', 'CNT_', 'CREDIT', 'INCOME',
                               'BUREAU_', 'PREV_', 'ANNUITY']

        for col in df.columns:
            if col in ['TARGET', 'SK_ID_CURR']:
                continue

            is_alternative = any(keyword in col.upper() for keyword in alternative_keywords)
            is_traditional = any(keyword in col.upper() for keyword in traditional_keywords)

            if is_alternative and not is_traditional:
                self.alternative_features.append(col)
            else:
                self.traditional_features.append(col)

        self.all_features = df.columns.tolist()

        return self.alternative_features, self.traditional_features

    def preprocess_and_save(self, train_paths, test_paths=None):
        """Main preprocessing function with windowizing"""
        print("\n" + "="*60)
        print("STEP 1: DATA PREPROCESSING")
        print("="*60)

        # Process training data
        print("\n1.1 Processing training data...")
        train_data, train_target = self.load_and_merge_data(train_paths)

        # Clean data
        print("1.2 Cleaning data...")
        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        train_data = train_data.fillna(0)

        # Apply windowizing - IMPORTANT: Do this BEFORE encoding
        print("1.3 WINDOWIZING - Power transformation for skewed features...")
        train_data = self.apply_windowizing(train_data)

        # Encode categorical
        print("1.4 Encoding categorical features...")
        train_data = self.encode_categorical(train_data)

        # Separate features
        print("1.5 Separating feature types...")
        alt_features, trad_features = self.separate_features(train_data)
        print(f"  Alternative features: {len(alt_features)}")
        print(f"  Traditional features: {len(trad_features)}")

        # Prepare for modeling
        X = train_data.drop(['SK_ID_CURR', 'TARGET'], axis=1, errors='ignore')
        y = train_target

        # Train-validation split
        print("1.6 Splitting train/validation...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Apply SMOTE with reduced ratio
        print("1.7 Applying SMOTE (50% ratio to save memory)...")
        smote = SMOTE(random_state=42, sampling_strategy=0.5)

        # Scale and balance for each feature set
        datasets = {}

        # All features
        scaler_all = StandardScaler()
        X_train_scaled = scaler_all.fit_transform(X_train)
        X_val_scaled = scaler_all.transform(X_val)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train.astype(int))

        datasets['all'] = {
            'X_train': X_train_balanced,
            'X_val': X_val_scaled,
            'y_train': y_train_balanced,
            'y_val': y_val,
            'features': X.columns.tolist(),
            'scaler': scaler_all
        }

        # Traditional features
        if len(trad_features) > 0:
            trad_cols = [col for col in X.columns if col in trad_features]
            if len(trad_cols) > 0:
                X_train_trad = X_train[trad_cols]
                X_val_trad = X_val[trad_cols]

                scaler_trad = StandardScaler()
                X_train_trad_scaled = scaler_trad.fit_transform(X_train_trad)
                X_val_trad_scaled = scaler_trad.transform(X_val_trad)
                X_train_trad_balanced, y_train_trad_balanced = smote.fit_resample(
                    X_train_trad_scaled, y_train.astype(int)
                )

                datasets['traditional'] = {
                    'X_train': X_train_trad_balanced,
                    'X_val': X_val_trad_scaled,
                    'y_train': y_train_trad_balanced,
                    'y_val': y_val,
                    'features': trad_cols,
                    'scaler': scaler_trad
                }

        # Alternative features
        if len(alt_features) > 0:
            alt_cols = [col for col in X.columns if col in alt_features]
            if len(alt_cols) > 0:
                X_train_alt = X_train[alt_cols]
                X_val_alt = X_val[alt_cols]

                scaler_alt = StandardScaler()
                X_train_alt_scaled = scaler_alt.fit_transform(X_train_alt)
                X_val_alt_scaled = scaler_alt.transform(X_val_alt)
                X_train_alt_balanced, y_train_alt_balanced = smote.fit_resample(
                    X_train_alt_scaled, y_train.astype(int)
                )

                datasets['alternative'] = {
                    'X_train': X_train_alt_balanced,
                    'X_val': X_val_alt_scaled,
                    'y_train': y_train_alt_balanced,
                    'y_val': y_val,
                    'features': alt_cols,
                    'scaler': scaler_alt
                }

        # Save preprocessed data
        print("\n1.8 Saving preprocessed data...")
        with open('preprocessed_data.pkl', 'wb') as f:
            pickle.dump(datasets, f)

        # Save preprocessor
        with open('preprocessor.pkl', 'wb') as f:
            pickle.dump(self, f)

        print("✅ Preprocessing complete! Saved to 'preprocessed_data.pkl'")
        print(f"\n   Dataset sizes:")
        for name, data in datasets.items():
            print(f"   - {name}: Train {data['X_train'].shape}, Val {data['X_val'].shape}")

        # Clear memory
        del train_data, X, y, X_train, X_val
        gc.collect()

        return datasets
