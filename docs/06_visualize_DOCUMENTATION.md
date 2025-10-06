# 06_Visualize Documentation

## Overview
This file provides functionality to visualize the performance of credit evaluation models. It represents thin-file customer analysis results and model comparison results through various charts and graphs to facilitate intuitive understanding. It generates professional visualizations using matplotlib.

## Main Functions

### 1. create_thin_file_plots(thin_file_results)
**Purpose**: Visualizes thin-file customer analysis results

**Parameters**:
- `thin_file_results`: List of thin-file analysis results

**Generated Visualizations**:

#### Plot 1: AUC Comparison (Thin-file vs Regular Customers)
- **Chart Type**: Bar Chart
- **X-axis**: Model names
- **Y-axis**: AUC scores
- **Colors**: 
  - Orange: Thin-file customers
  - Blue: Regular customers
- **Features**: Compare performance differences between two groups at a glance

#### Plot 2: Acceptance Rate Comparison (5% Bad Rate Standard)
- **Chart Type**: Bar Chart
- **X-axis**: Model names
- **Y-axis**: Acceptance rate (%)
- **Colors**: 
  - Orange: Thin-file customers
  - Blue: Regular customers
- **Features**: Practical performance comparison from business perspective

**Visualization Features**:
- 1 row × 2 column subplot structure (figsize=(12, 5))
- Grid background (alpha=0.3) for improved readability
- 45-degree rotation of model names for space efficiency
- Clear distinction with legends and titles

**Saved files**: `thin_file_analysis.png`

### 2. create_comparison_plots(df_results)
**Purpose**: Comprehensively visualizes model comparison results

**Parameters**:
- `df_results`: Model results DataFrame

**Generated Visualizations** (2 rows × 2 columns subplot):

#### Plot 1: Model-wise Feature Set AUC Comparison
- **Chart Type**: Grouped Bar Chart
- **X-axis**: Model names
- **Y-axis**: AUC scores
- **Colors**: Distinguished by feature set (traditional, alternative, all)
- **Features**: Compare performance by feature set for each model

#### Plot 2: Model-wise Acceptance Rate Comparison
- **Chart Type**: Grouped Bar Chart
- **X-axis**: Model names
- **Y-axis**: Acceptance rate
- **Colors**: Distinguished by feature set
- **Features**: Model performance comparison from business perspective

#### Plot 3: Average Performance by Feature Set
- **Chart Type**: Bar Chart
- **X-axis**: Feature sets (traditional, alternative, all)
- **Y-axis**: Average scores
- **Colors**: AUC (blue), Acceptance rate (orange)
- **Features**: Compare overall effectiveness of feature sets

#### Plot 4: Top Model Configurations (Top 5)
- **Chart Type**: Horizontal Bar Chart
- **Y-axis**: Top models (maximum 5)
- **X-axis**: AUC scores
- **Colors**: Distinguished by feature set
  - Blue: all
  - Green: traditional  
  - Orange: alternative
- **Features**: Identify highest performing model configurations

**Visualization Features**:
- 2 rows × 2 columns subplot structure (figsize=(14, 10))
- Grid background applied to all charts
- 45-degree rotation of model names for space efficiency
- Dynamic data processing (handles cases with no data)

**Saved files**: `model_comparison.png`

## Data Processing Logic

### 1. Thin-file Result Processing
```python
df = pd.DataFrame(thin_file_results)
```
- Convert list-form results to DataFrame
- Organize performance metrics by model and feature set

### 2. Model Comparison Result Processing
```python
# Create pivot table
pivot_auc = df_results.pivot_table(
    index='model_name', 
    columns='feature_set',
    values='auc_score', 
    aggfunc='mean'
)
```
- Create pivot table based on model and feature set
- Aggregate with mean values for stable visualization

### 3. Top Model Selection
```python
n_top = min(5, len(df_results))
top_models = df_results.nlargest(n_top, 'auc_score')
```
- Select top 5 models based on AUC
- Dynamic adjustment when data is insufficient

## Visualization Style

### Color Scheme
- **Orange**: Thin-file customers, acceptance rate
- **Blue**: Regular customers, AUC, All features
- **Green**: Traditional features
- **Gray**: Default, others

### Layout Features
- **Grid Background**: Apply alpha=0.3 grid to all charts
- **Rotated Labels**: 45-degree rotation of X-axis labels for improved readability
- **Legends**: Clear legends to distinguish data
- **Titles**: Clearly express the purpose of each chart

### File Save Settings
- **Resolution**: 100 DPI
- **Margins**: Optimized with bbox_inches='tight'
- **Format**: PNG format

## Usage Example

```python
# Visualize thin-file analysis results
thin_file_results = [
    {
        'model_name': 'LightGBM',
        'auc_thin_file': 0.7234,
        'auc_regular': 0.7891,
        'acceptance_thin_file': 0.45,
        'acceptance_regular': 0.62
    },
    # ... other models
]

create_thin_file_plots(thin_file_results)

# Visualize model comparison results
df_results = pd.DataFrame([
    {
        'model_name': 'LightGBM',
        'feature_set': 'traditional',
        'auc_score': 0.7234,
        'acceptance_rate': 0.45
    },
    # ... other results
])

create_comparison_plots(df_results)
```

## Output Example

### Console Output
```
📊 Thin-file visualization saved to 'thin_file_analysis.png'
📊 Visualization saved to 'model_comparison.png'
```

### Generated Files
1. **thin_file_analysis.png**: Thin-file customer analysis visualization
2. **model_comparison.png**: Comprehensive model comparison visualization

## Visualization Significance

### 1. Business Insights
- **Intuitive Understanding**: Visual representation of complex numerical data
- **Decision Support**: Provide clear criteria for model selection
- **Performance Reporting**: Effective result delivery to stakeholders

### 2. Technical Analysis
- **Performance Comparison**: Clearly identify performance differences between models
- **Feature Effects**: Visualize contribution by feature set
- **Optimization Direction**: Identify areas needing improvement

### 3. Research Value
- **Reproducibility**: Maintain consistency with identical visualization styles
- **Scalability**: Easy expansion when adding new models or features
- **Standardization**: Provide standard visualization for credit evaluation model comparison

## Dependencies
- matplotlib.pyplot: Visualization generation
- pandas: Data processing and pivot table creation
- src.utils.paths.fig_path: Figure file path management

## Notes
- Appropriate handling required when data is unavailable
- Visualization time may be long for large datasets
- Consider colorblind users when color distinction is important
- File size may be large for high-resolution output

## File Information
- **Original File**: 06_visualize.ipynb
- **Created**: Automatically generated in Colab
- **Lines of Code**: Approximately 120 lines
- **Main Functions**: 2 (create_thin_file_plots, create_comparison_plots)
- **Generated Images**: 2 (PNG format)
- **Main Functionality**: Model performance visualization and analysis
