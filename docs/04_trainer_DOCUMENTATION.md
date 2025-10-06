# 04_Trainer Documentation

## Overview
This file provides a comprehensive training system that sequentially trains and evaluates various machine learning models. It supports 11 different models and allows users to select and train their desired models. It includes special analysis capabilities for "thin-file" customers (customers with limited credit history), which is crucial in credit evaluation.

## Main Classes

### SequentialModelTrainer
**Purpose**: An integrated training system that sequentially trains multiple machine learning models and compares their performance

**Key Features**:
- Supports 11 different models
- User-selection based model training
- Multi-feature set support (traditional, alternative, all)
- Special analysis for thin-file customers
- Automatic model and result saving

## Supported Models

### 1. Linear Models
- **Linear_Regression**: Linear regression-based classifier
- **Logistic_Regression**: Logistic regression (max_iter=1000)

### 2. Tree-based Models
- **Decision_Tree**: Decision tree (max_depth=10)
- **Random_Forest**: Random forest (n_estimators=100, max_depth=10)
- **Gradient_Boosting**: Gradient boosting (n_estimators=100, max_depth=5)
- **LightGBM**: LightGBM classifier (n_estimators=100)
- **Extra_Trees**: Extra trees (n_estimators=100, max_depth=10)

### 3. Deep Learning Models
- **TabNet**: Custom TabNet implementation (n_steps=3, feature_dim=8, epochs=30)
- **Wide_Deep**: Wide & Deep network (deep_layers=[128, 64, 32], epochs=30)
- **MLP**: Multi-layer perceptron (hidden_layer_sizes=(100, 50), max_iter=500)

### 4. Other Models
- **SVM**: SGD-based SVM (loss='log_loss', class_weight='balanced')

## Core Methods

### 1. get_all_models()
**Function**: Defines all available models
**Returns**: Dictionary containing model names and instances
**Features**: Organizes 11 models by category

### 2. select_models(selection=None)
**Function**: Allows users to select models for training
**Parameters**:
- `selection`: Model selection string (can also be set via MODEL_SELECTION environment variable)

**Selection Options**:
- Individual model numbers (1-11)
- Quick options:
  - `0`: All models (11 models)
  - `99`: Quick mode (LightGBM, Random_Forest, Logistic_Regression)
  - `88`: Deep learning only (TabNet, Wide_Deep, MLP)
  - `77`: Traditional ML only (excluding deep learning)

**Returns**: Dictionary of selected models

### 3. calculate_acceptance_rate(y_true, y_pred_proba, target_bad_rate=0.05)
**Function**: Calculates acceptance rate at a fixed bad rate
**Parameters**:
- `y_true`: True labels
- `y_pred_proba`: Prediction probabilities
- `target_bad_rate`: Target bad rate (default: 5%)

**How it works**:
1. Sort customers by prediction probability (lower probability = better customer)
2. Gradually increase acceptance threshold
3. Find the point where actual bad rate matches target bad rate
4. Return acceptance rate at that point

**Returns**: Dictionary containing acceptance rate, threshold, and target/actual bad rates

### 4. train_single_model(model_name, model, datasets)
**Function**: Trains a single model on all feature sets
**Parameters**:
- `model_name`: Model name
- `model`: Model instance
- `datasets`: Feature set-specific data

**Process**:
1. Copy and train model for each feature set
2. Predict and evaluate on validation data
3. Calculate AUC score and acceptance rate
4. Save results and model files
5. Compare performance across feature sets

**Saved files**: `models/{model_name}_{feature_set}_model.pkl`

### 5. analyze_thin_file_customers(datasets, model_results)
**Function**: Special analysis for thin-file customers
**Parameters**:
- `datasets`: Feature set-specific data
- `model_results`: Model training results

**Thin-file customer identification methods**:
- **Traditional/All features**: Customers with very low Bureau/credit feature values
- **Alternative features**: Customers with low feature standard deviation (indicator of new customers)

**Analysis content**:
1. Performance comparison between thin-file vs regular customers
2. Thin-file customer performance by feature set
3. Effect of alternative data on thin-file customers
4. Finding optimal model configuration

**Saved files**: `thin_file_analysis_by_features.csv`

### 6. compare_feature_sets(model_results)
**Function**: Performance comparison between feature sets
**Comparison items**: AUC score, acceptance rate

### 7. save_results()
**Function**: Saves all results to CSV file
**Saved files**: `model_results.csv`

## Usage Example

```python
# Initialize training system
trainer = SequentialModelTrainer()

# Select models (user input or environment variable)
selected_models = trainer.select_models("99")  # Quick mode

# Prepare datasets (traditional, alternative, all feature sets)
datasets = {
    'traditional': {'X_train': X_trad_train, 'y_train': y_train, ...},
    'alternative': {'X_train': X_alt_train, 'y_train': y_train, ...},
    'all': {'X_train': X_all_train, 'y_train': y_train, ...}
}

# Train each model
for model_name, model in selected_models.items():
    results = trainer.train_single_model(model_name, model, datasets)

# Analyze thin-file customers
thin_file_results = trainer.analyze_thin_file_customers(datasets, trainer.results)

# Save results
df_results = trainer.save_results()
```

## Output Information

### Training Process Output
- Training progress by model
- Performance metrics by feature set
- AUC scores and acceptance rates
- Comparison results between feature sets

### Thin-file Analysis Output
- Distribution of thin-file vs regular customers
- Thin-file customer performance for each model
- Analysis of alternative data effectiveness
- Recommendations for optimal model configuration

### Saved Files
- `model_results.csv`: Overall model performance results
- `thin_file_analysis_by_features.csv`: Detailed thin-file customer analysis
- `models/`: Trained model files

## Dependencies
- numpy, pandas
- scikit-learn (various models)
- lightgbm
- tensorflow/keras (for deep learning models)
- joblib (model saving)
- gc (memory management)

## Environment Variables
- `MODEL_SELECTION`: Environment variable for non-interactive model selection

## Notes
- Deep learning models may have high memory usage
- Thin-file customer identification is heuristic-based
- Sufficient disk space required for model saving
- Training time may be long for large datasets

## File Information
- **Original File**: 04_Trainer.ipynb
- **Created**: Automatically generated in Colab
- **Lines of Code**: Approximately 440 lines
- **Main Classes**: 1 (SequentialModelTrainer)
- **Supported Models**: 11
- **Main Functionality**: Integrated model training and evaluation system
