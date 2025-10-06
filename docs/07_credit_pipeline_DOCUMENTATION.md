# 07_Credit_Pipeline Documentation

## Overview
This file provides a comprehensive credit risk modeling pipeline that encapsulates the entire workflow from data preprocessing to model evaluation and visualization. The `CreditRiskPipeline` class orchestrates all components of the credit risk assessment process, including data preprocessing, model selection, training, analysis, and visualization in a sequential manner.

## Main Classes

### CreditRiskPipeline
**Purpose**: A complete credit risk modeling pipeline that integrates preprocessing, model selection, training, analysis, and visualization

**Key Features**:
- End-to-end pipeline management
- Sequential execution of all pipeline components
- Caching and reprocessing options for data preprocessing
- Comprehensive model performance analysis
- Thin-file customer analysis
- Automated visualization generation
- Non-interactive execution support via environment variables

## Core Methods

### 1. preprocess(train_paths, test_paths=None, reprocess_choice=None)
**Function**: Handles data preprocessing with caching and reprocessing options

**Parameters**:
- `train_paths`: Dictionary of training data paths
- `test_paths`: Optional dictionary of test data paths
- `reprocess_choice`: Optional string ('y'/'n') to control reprocessing without user prompt

**How it works**:
1. Checks for existing preprocessed data cache (`preprocessed_data.pkl`)
2. If cache exists, loads cached data and optionally prompts for reprocessing
3. If no cache or reprocessing requested, runs full preprocessing pipeline
4. Supports non-interactive mode via `REPROCESS` environment variable

**Returns**: Preprocessed datasets dictionary

### 2. select_models(selection=None)
**Function**: Delegates model selection to the SequentialModelTrainer

**Parameters**:
- `selection`: Optional model selection string (e.g., '99' for quick mode)

**Returns**: Tuple of (trainer_instance, selected_models_dictionary)

### 3. train_models(trainer, selected_models, datasets)
**Function**: Trains selected models on all feature sets with progress tracking

**Parameters**:
- `trainer`: SequentialModelTrainer instance
- `selected_models`: Dictionary of selected model instances
- `datasets`: Preprocessed datasets

**Process**:
1. Iterates through selected models with progress tracking
2. Trains each model on all feature sets (traditional, alternative, all)
3. Saves intermediate results after each model
4. Performs garbage collection to manage memory
5. Displays completion status for each model

**Returns**: List of all training results

### 4. analyze_thin_file(trainer, datasets, all_results)
**Function**: Analyzes performance for thin-file customers

**Parameters**:
- `trainer`: SequentialModelTrainer instance
- `datasets`: Preprocessed datasets
- `all_results`: List of all model training results

**Returns**: Thin-file analysis results

### 5. compare_alternative_impact(all_results, datasets)
**Function**: Compares alternative vs traditional features for thin-file customers

**Parameters**:
- `all_results`: List of all model training results
- `datasets`: Preprocessed datasets

**Returns**: Alternative data impact analysis results

### 6. create_thin_file_plots(thin_file_results)
**Function**: Generates and saves thin-file analysis visualizations

**Parameters**:
- `thin_file_results`: Results from thin-file analysis

**Returns**: Visualization output

### 7. create_comparison_plots(df_results)
**Function**: Generates and saves overall model comparison visualizations

**Parameters**:
- `df_results`: DataFrame of all model results

**Returns**: Visualization output

### 8. run(train_paths, test_paths=None, selection=None, reprocess_choice=None)
**Function**: Executes the complete sequential pipeline

**Parameters**:
- `train_paths`: Dictionary of training data paths
- `test_paths`: Optional dictionary of test data paths
- `selection`: Non-interactive model selection string (e.g., '99')
- `reprocess_choice`: 'y'/'n' to control reprocessing without prompt

**Pipeline Execution Flow**:
1. **Data Preprocessing**: Load or preprocess data with caching
2. **Model Selection**: Interactive or non-interactive model selection
3. **Model Training**: Train selected models on all feature sets
4. **Thin-file Analysis**: Analyze performance for thin-file customers
5. **Alternative Data Impact**: Compare feature set effectiveness
6. **Results Summary**: Display top 10 models and best performers
7. **Visualization**: Generate comparison and thin-file plots

**Output Summary**:
- Top 10 model performances (overall)
- Best model overall with AUC and acceptance rate
- Best model for thin-file customers
- Comprehensive visualizations

**Returns**: DataFrame of all results sorted by AUC score

## Pipeline Architecture

### Sequential Execution Flow
```
Data Preprocessing → Model Selection → Model Training → 
Thin-file Analysis → Alternative Impact Analysis → 
Results Summary → Visualization
```

### Component Integration
- **DataPreprocessor**: Handles data loading, cleaning, and feature engineering
- **SequentialModelTrainer**: Manages model selection and training
- **Analysis Functions**: Compare alternative data effectiveness
- **Visualization Functions**: Generate comprehensive plots

### Caching Strategy
- Preprocessed data cached in `preprocessed_data.pkl`
- Model results saved incrementally during training
- Visualization files saved as PNG images

## Environment Variables

### Non-Interactive Execution
- `REPROCESS`: Controls data reprocessing ('y'/'n')
- `MODEL_SELECTION`: Controls model selection (e.g., '99', '88', '77')

## Usage Example

### Interactive Mode
```python
# Initialize pipeline
pipeline = CreditRiskPipeline()

# Define data paths
train_paths = {
    'application': 'data/application_train.csv',
    'bureau': 'data/bureau.csv',
    'previous': 'data/previous_application.csv'
}

# Run complete pipeline
results_df = pipeline.run(train_paths)
```

### Non-Interactive Mode
```python
import os

# Set environment variables for non-interactive execution
os.environ['MODEL_SELECTION'] = '99'  # Quick mode
os.environ['REPROCESS'] = 'n'         # Use cached data

# Run pipeline
pipeline = CreditRiskPipeline()
results_df = pipeline.run(train_paths)
```

### Step-by-Step Execution
```python
# Execute pipeline steps individually
pipeline = CreditRiskPipeline()

# Step 1: Preprocess data
datasets = pipeline.preprocess(train_paths)

# Step 2: Select models
trainer, selected_models = pipeline.select_models('99')

# Step 3: Train models
all_results = pipeline.train_models(trainer, selected_models, datasets)

# Step 4: Analyze thin-file customers
thin_file_results = pipeline.analyze_thin_file(trainer, datasets, all_results)

# Step 5: Compare alternative impact
pipeline.compare_alternative_impact(all_results, datasets)

# Step 6: Create visualizations
df_results = pd.DataFrame(all_results)
pipeline.create_comparison_plots(df_results)
pipeline.create_thin_file_plots(thin_file_results)
```

## Output Information

### Console Output
- Pipeline execution progress with clear section headers
- Model training progress with completion status
- Top 10 model performances summary
- Best model recommendations (overall and thin-file)
- Alternative data impact analysis results

### Generated Files
- `preprocessed_data.pkl`: Cached preprocessed data
- `model_results.csv`: Complete model performance results
- `thin_file_analysis_by_features.csv`: Detailed thin-file analysis
- `alternative_impact_thin_file.csv`: Alternative data impact analysis
- `models/`: Directory containing trained model files
- `thin_file_analysis.png`: Thin-file customer analysis visualization
- `model_comparison.png`: Comprehensive model comparison visualization

### Results DataFrame
**Columns**:
- `model_name`: Name of the trained model
- `feature_set`: Feature set used (traditional/alternative/all)
- `auc_score`: Area Under Curve score
- `acceptance_rate`: Acceptance rate at 5% bad rate
- `threshold`: Classification threshold used
- `actual_bad_rate`: Actual bad rate achieved
- `n_features`: Number of features used
- `model`: Trained model instance

## Dependencies
- os, gc, pickle: System and memory management
- numpy, pandas: Data manipulation and analysis
- src.pipeline.data_preprocessor: Data preprocessing functionality
- src.pipeline.trainer: Model training functionality
- src.pipeline.analysis: Analysis functions
- src.pipeline.visualize: Visualization functions
- src.utils.paths: Path management utilities

## Key Features

### 1. Comprehensive Pipeline
- Integrates all components of credit risk modeling
- Sequential execution with clear progress tracking
- End-to-end automation with manual override options

### 2. Flexible Execution
- Interactive mode for exploration and experimentation
- Non-interactive mode for automated execution
- Step-by-step execution for debugging and customization

### 3. Performance Optimization
- Data caching to avoid redundant preprocessing
- Incremental result saving during training
- Memory management with garbage collection

### 4. Comprehensive Analysis
- Overall model performance comparison
- Thin-file customer specific analysis
- Alternative data effectiveness evaluation
- Automated visualization generation

### 5. Production Ready
- Environment variable support for automation
- Error handling and graceful degradation
- Comprehensive logging and progress tracking

## Notes
- Pipeline assumes proper data structure and file paths
- Memory usage can be high with large datasets and multiple models
- Visualization files are saved in current working directory
- Model files are saved in `models/` subdirectory
- Results are automatically sorted by AUC score for easy interpretation

## File Information
- **Original File**: 07_Credit_Pipeline.ipynb
- **Created**: Automatically generated in Colab
- **Lines of Code**: Approximately 160 lines
- **Main Classes**: 1 (CreditRiskPipeline)
- **Main Methods**: 8 core methods
- **Main Functionality**: Complete credit risk modeling pipeline orchestration
