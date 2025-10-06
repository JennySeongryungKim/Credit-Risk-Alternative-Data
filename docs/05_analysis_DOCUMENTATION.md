# 05_Analysis Documentation

## Overview
This file provides functionality to analyze the performance of credit evaluation models and specifically compare the effectiveness of alternative data for "thin-file" customers (customers with limited credit history). It quantitatively measures and evaluates the performance differences between traditional credit data and alternative data.

## Main Functions

### compare_alternative_impact(all_results, datasets, trainer)
**Purpose**: Compares and analyzes the impact of alternative data on thin-file customers

**Parameters**:
- `all_results`: List of training results for all models
- `datasets`: Dictionary of datasets by feature set
- `trainer`: Trainer instance (currently unused)

**How it works**:

#### 1. Thin-file Customer Identification
```python
feature_variance = np.var(X_val_all, axis=1)
threshold = np.percentile(feature_variance, 20)
thin_file_mask = feature_variance < threshold
```
- Classifies the bottom 20% as thin-file customers based on feature variance
- Assumes customers with low feature variance = customers with limited credit history

#### 2. Model-wise Feature Set Grouping
- Groups results by model name
- Separates traditional, alternative, and all feature set results for each model

#### 3. Performance Comparison Analysis
For each model, calculates:
- **Traditional Feature AUC**: Using only traditional credit data
- **Alternative Feature AUC**: Using only alternative data  
- **All Feature AUC**: Using all combined data
- **Improvement**: Alternative vs Traditional AUC difference

#### 4. Result Output
```
📊 Comparing Feature Sets for Thin-File Customers:
--------------------------------------------------

Model_Name:
  Traditional AUC: 0.7234
  Alternative AUC: 0.7891
  All features AUC: 0.8123
  Alternative improvement: +0.0657 (+9.1%)
```

#### 5. Result Saving
- `alternative_impact_thin_file.csv`: Saves detailed comparison results
- Outputs summary of average and maximum improvements

## Core Analysis Logic

### 1. Thin-file Customer Definition
- **Criteria**: Bottom 20% of feature variance
- **Assumption**: Customers with low feature variance = customers with limited credit history
- **Purpose**: Measure the effectiveness of alternative data for customers lacking credit history

### 2. Feature Set Comparison
- **Traditional**: Traditional credit bureau data, loan history, etc.
- **Alternative**: Alternative data (social media, transaction patterns, etc.)
- **All**: Combined data from all features

### 3. Performance Metrics
- **AUC (Area Under Curve)**: Area under the ROC curve
- **Improvement**: Alternative AUC - Traditional AUC
- **Improvement Rate**: (Improvement / Traditional AUC) × 100

### 4. Statistical Summary
- Calculate average improvement
- Identify model with maximum improvement
- Evaluate overall effectiveness of alternative data

## Output Example

```
📊 Comparing Feature Sets for Thin-File Customers:
--------------------------------------------------

LightGBM:
  Traditional AUC: 0.7234
  Alternative AUC: 0.7891
  All features AUC: 0.8123
  Alternative improvement: +0.0657 (+9.1%)

Random_Forest:
  Traditional AUC: 0.7156
  Alternative AUC: 0.7745
  All features AUC: 0.7989
  Alternative improvement: +0.0589 (+8.2%)

💾 Alternative data impact saved to 'alternative_impact_thin_file.csv'

📊 SUMMARY:
  Average improvement from alternative data: +0.0623
  Best improvement: +0.0657 (LightGBM)
```

## Saved Files

### alternative_impact_thin_file.csv
**Column Structure**:
- `model`: Model name
- `auc_traditional`: Traditional feature AUC
- `auc_alternative`: Alternative feature AUC  
- `auc_all`: All feature AUC
- `improvement`: Improvement (Alternative - Traditional)

## Usage Example

```python
# Prepare model training results and datasets
all_results = [
    {'model_name': 'LightGBM', 'feature_set': 'traditional', 'model': model1, ...},
    {'model_name': 'LightGBM', 'feature_set': 'alternative', 'model': model2, ...},
    {'model_name': 'LightGBM', 'feature_set': 'all', 'model': model3, ...},
    # ... other models
]

datasets = {
    'all': {'X_val': X_all_val, 'y_val': y_val},
    'traditional': {'X_val': X_trad_val, 'y_val': y_val},
    'alternative': {'X_val': X_alt_val, 'y_val': y_val}
}

# Execute alternative data effectiveness analysis
compare_alternative_impact(all_results, datasets, trainer)
```

## Analysis Significance

### 1. Business Value
- **New Customer Expansion**: Evaluate customers without credit history using alternative data
- **Risk Management**: Reduce bad rates through more accurate credit evaluation
- **Profitability Enhancement**: Safely approve more customers

### 2. Technical Value
- **Feature Importance**: Understand which features are more effective for thin-file customers
- **Model Selection**: Identify models optimized for thin-file customers
- **Data Strategy**: Determine priorities for alternative data collection

### 3. Research Value
- **Alternative Credit Assessment**: Overcome limitations of traditional credit assessment
- **Financial Inclusion**: Improve financial accessibility for those without credit history
- **Data Fusion**: Effective combination methods for diverse data sources

## Dependencies
- numpy: Numerical calculations
- pandas: Data processing and storage
- sklearn.metrics.roc_auc_score: AUC calculation
- src.utils.paths.artifact_path: File path management

## Notes
- Thin-file customer identification is heuristic-based, requiring domain knowledge
- Feature variance criteria may not be suitable for all situations
- Quality of alternative data significantly impacts results
- Analysis results may be specific to particular datasets

## File Information
- **Original File**: 05_Analysis.ipynb
- **Created**: Automatically generated in Colab
- **Lines of Code**: Approximately 90 lines
- **Main Functions**: 1 (compare_alternative_impact)
- **Main Functionality**: Alternative data effectiveness analysis
- **Analysis Target**: Credit evaluation performance for thin-file customers
