# 03_Customer_Models Documentation

## Overview
This file defines custom machine learning models for credit evaluation and customer classification. It includes both traditional machine learning and deep learning models, implemented to be compatible with scikit-learn pipelines by inheriting from `BaseEstimator` and `ClassifierMixin`.

## Main Classes

### 1. TabNetClassifier
**Purpose**: An attention-based deep learning model for tabular data

**Key Features**:
- Simplified implementation of Google’s TabNet architecture
- Feature selection through an attention mechanism
- Specialized for binary classification

**Key Parameters**:
- `n_steps`: Number of attention steps (default: 3)
- `feature_dim`: Feature dimension (default: 8)
- `learning_rate`: Learning rate (default: 0.02)
- `batch_size`: Batch size (default: 256)
- `epochs`: Number of epochs (default: 50)

**Architecture**:
1. Input feature transformation (Dense + BatchNormalization)
2. Feature selection via attention mechanism
3. Transformation of selected features with dropout
4. Binary classification output with sigmoid activation

**Main Methods**:
- `_build_model()`: Constructs the TabNet architecture
- `fit()`: Trains the model (includes EarlyStopping and ReduceLROnPlateau callbacks)
- `predict_proba()`: Predicts probabilities (with error handling)
- `predict()`: Predicts class labels

### 2. WideDeepClassifier
**Purpose**: Binary classification using a Wide & Deep network

**Key Features**:
- Wide component: Linear model (captures feature interactions)
- Deep component: Deep neural network (learns complex patterns)
- Combines both components for comprehensive predictions

**Key Parameters**:
- `deep_layers`: Structure of the deep network layers (default: [128, 64, 32])
- `learning_rate`: Learning rate (default: 0.001)
- `batch_size`: Batch size (default: 256)
- `epochs`: Number of epochs (default: 50)

**Architecture**:
1. Wide component: Direct linear transformation of inputs
2. Deep component: Multi-layer neural network (Dense + BatchNormalization + Dropout)
3. Combines Wide and Deep outputs using an Add layer
4. Final prediction with sigmoid activation

**Main Methods**:
- `_build_model()`: Constructs the Wide & Deep architecture
- `fit()`: Trains the model
- `predict_proba()`: Predicts probabilities
- `predict()`: Predicts class labels

### 3. LinearRegressionClassifier
**Purpose**: A wrapper class that adapts linear regression for binary classification

**Key Features**:
- Wraps scikit-learn’s `LinearRegression`
- Converts continuous outputs to probabilities
- Automatically searches for the optimal classification threshold

**Key Parameters**:
- `threshold`: Classification threshold (automatically determined during training)

**How It Works**:
1. Predicts continuous values using linear regression
2. Searches for the optimal threshold from training data (over 100 intervals)
3. Clips predictions to the [0, 1] range
4. Performs binary classification based on the threshold

**Main Methods**:
- `fit()`: Trains the linear regression model and finds the optimal threshold
- `predict_proba()`: Predicts probabilities (with clipping)
- `predict()`: Predicts class labels based on the threshold

## Common Interface

### Scikit-learn Compatibility
All classes implement scikit-learn’s standard interface:
- `get_params()`: Returns hyperparameters
- `set_params()`: Sets hyperparameters
- `fit()`: Trains the model
- `predict()`: Predicts class labels
- `predict_proba()`: Predicts probabilities

### Error Handling
- Fallback to random predictions if deep learning model predictions fail
- Safe handling of empty inputs
- Detailed error messages for exceptions

## Usage Example

```python
# Using TabNetClassifier
tabnet = TabNetClassifier(n_steps=5, feature_dim=16, epochs=100)
tabnet.fit(X_train, y_train)
predictions = tabnet.predict(X_test)
probabilities = tabnet.predict_proba(X_test)

# Using WideDeepClassifier
wide_deep = WideDeepClassifier(deep_layers=[256, 128, 64], epochs=50)
wide_deep.fit(X_train, y_train)
predictions = wide_deep.predict(X_test)

# Using LinearRegressionClassifier
linear_clf = LinearRegressionClassifier()
linear_clf.fit(X_train, y_train)
predictions = linear_clf.predict(X_test)
```

## Dependencies
- numpy
- scikit-learn
- tensorflow/keras
- typing (optional)

## Notes
- Deep learning models may perform better with GPU support
- TabNet and WideDeep may have high memory usage
- LinearRegressionClassifier is fast but may have limited performance
- All models are specialized for binary classification

## File Information
- **Original File**: 03_Custom_Models.ipynb
- **Created**: Automatically generated in Colab
- **Lines of Code**: Approximately 270 lines
- **Number of Classes**: 3
- **Main Functionality**: Defines custom machine learning models

