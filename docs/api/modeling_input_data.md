# modeling_input_data

Core data structures for handling input data preprocessing and validation in tfbpmodeling.

::: tfbpmodeling.modeling_input_data

## Overview

The `modeling_input_data` module provides the fundamental `ModelingInputData` class that handles:

- **Data loading**: Reading CSV files for response and predictor data
- **Validation**: Ensuring data consistency and format compliance
- **Preprocessing**: Feature filtering, normalization, and transformation
- **Integration**: Merging response and predictor data for modeling

This class serves as the foundation for all downstream modeling operations.

## Core Classes

### ModelingInputData

The primary class for managing input data throughout the modeling workflow.

#### Key Features

- **Automatic data validation**: Checks file formats, column consistency, and data types
- **Feature filtering**: Removes blacklisted genes and handles missing data
- **Index alignment**: Ensures consistent gene identifiers between response and predictor files
- **Data integration**: Combines multiple data sources into modeling-ready format

#### Initialization

```python
from tfbpmodeling.modeling_input_data import ModelingInputData

# Basic initialization
data = ModelingInputData(
    response_file='expression.csv',
    predictors_file='binding.csv',
    perturbed_tf='YPD1'
)

# With optional parameters
data = ModelingInputData(
    response_file='expression.csv',
    predictors_file='binding.csv',
    perturbed_tf='YPD1',
    blacklist_file='exclude_genes.txt',
    normalize_weights=True,
    scale_by_std=True
)
```

#### Key Methods

##### Data Loading and Validation
- **`load_data()`**: Read and validate input CSV files
- **`validate_format()`**: Check data format compliance
- **`check_consistency()`**: Verify response-predictor alignment

##### Data Preprocessing
- **`filter_features()`**: Remove blacklisted and invalid features
- **`normalize_data()`**: Apply scaling and normalization
- **`handle_missing()`**: Deal with missing values appropriately

##### Data Access
- **`get_response_data()`**: Access processed response data
- **`get_predictor_data()`**: Access processed predictor data
- **`get_feature_names()`**: Retrieve feature identifiers
- **`get_sample_names()`**: Retrieve sample identifiers

## Data Format Requirements

### Response File Format

The response file must be a CSV with specific structure:

```csv
gene_id,sample1,sample2,sample3,sample4
YPD1,0.23,-1.45,0.87,-0.12
YBR123W,1.34,0.56,-0.23,0.78
YCR456X,-0.45,0.12,1.23,-0.56
```

**Requirements**:
- First column: Gene identifiers (must match predictor file)
- Subsequent columns: Numeric expression values
- Column names: Sample identifiers
- Must contain column matching `perturbed_tf` parameter

### Predictor File Format

The predictor file structure:

```csv
gene_id,TF1,TF2,TF3,TF4
YPD1,0.34,0.12,0.78,0.01
YBR123W,0.89,0.45,0.23,0.67
YCR456X,0.12,0.78,0.34,0.90
```

**Requirements**:
- First column: Gene identifiers (must match response file)
- Subsequent columns: Numeric binding values
- Column names: Transcription factor identifiers
- All values must be numeric (no missing values in binding data)

### Blacklist File Format

Optional exclusion file:

```
YBR999W
YCR888X
control_gene
technical_artifact
```

**Requirements**:
- Plain text file
- One gene identifier per line
- Gene IDs must match those in data files
- Comments not supported

## Usage Examples

### Basic Data Loading

```python
from tfbpmodeling.modeling_input_data import ModelingInputData

# Load data with minimal configuration
data = ModelingInputData(
    response_file='data/expression.csv',
    predictors_file='data/binding.csv',
    perturbed_tf='YPD1'
)

# Access processed data
response_data = data.get_response_data()
predictor_data = data.get_predictor_data()
feature_names = data.get_feature_names()

print(f"Loaded {len(feature_names)} features")
print(f"Response data shape: {response_data.shape}")
print(f"Predictor data shape: {predictor_data.shape}")
```

### Data Preprocessing Options

```python
# Advanced preprocessing
data = ModelingInputData(
    response_file='data/expression.csv',
    predictors_file='data/binding.csv',
    perturbed_tf='YPD1',
    blacklist_file='data/exclude_genes.txt',
    normalize_weights=True,
    scale_by_std=True,
    handle_missing='drop'  # or 'impute', 'zero'
)

# Check data quality
print(f"Original features: {data.original_feature_count}")
print(f"Filtered features: {len(data.get_feature_names())}")
print(f"Excluded features: {data.excluded_feature_count}")
```

### Integration with Modeling Pipeline

```python
# Prepare data for bootstrap modeling
from tfbpmodeling.bootstrapped_input_data import BootstrappedModelingInputData

# Base data
base_data = ModelingInputData(
    response_file='expression.csv',
    predictors_file='binding.csv',
    perturbed_tf='YPD1'
)

# Create bootstrap version
bootstrap_data = BootstrappedModelingInputData(
    base_data=base_data,
    n_bootstraps=1000,
    random_state=42
)
```

## Data Validation

### Automatic Checks

The class performs comprehensive validation:

```python
# File existence and readability
assert os.path.exists(response_file), f"Response file not found: {response_file}"
assert os.path.exists(predictors_file), f"Predictor file not found: {predictors_file}"

# Data format validation
assert response_df.index.equals(predictor_df.index), "Gene indices must match"
assert perturbed_tf in response_df.columns, f"Perturbed TF '{perturbed_tf}' not found"

# Data type validation
assert response_df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
assert predictor_df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
```

### Custom Validation

```python
# Add custom validation rules
def validate_expression_range(data):
    \"\"\"Ensure expression values are in reasonable range\"\"\"
    assert data.abs().max().max() < 10, "Expression values seem too large"

def validate_binding_range(data):
    \"\"\"Ensure binding values are probabilities\"\"\"
    assert (data >= 0).all().all(), "Binding values must be non-negative"
    assert (data <= 1).all().all(), "Binding values must be <= 1"

# Apply custom validation
data = ModelingInputData(
    response_file='expression.csv',
    predictors_file='binding.csv',
    perturbed_tf='YPD1',
    custom_validators=[validate_expression_range, validate_binding_range]
)
```

## Error Handling

### Common Errors and Solutions

#### File Format Errors
```python
# CSV parsing errors
try:
    data = ModelingInputData(response_file='malformed.csv', ...)
except pd.errors.ParserError as e:
    print(f"CSV format error: {e}")
    # Solution: Check file encoding, delimiters, quotes
```

#### Data Consistency Errors
```python
# Mismatched gene indices
try:
    data = ModelingInputData(...)
except ValueError as e:
    if "Gene indices must match" in str(e):
        print("Response and predictor files have different gene sets")
        # Solution: Align gene lists or use intersection
```

#### Missing Data Errors
```python
# Perturbed TF not found
try:
    data = ModelingInputData(perturbed_tf='MISSING_TF', ...)
except KeyError as e:
    print(f"Perturbed TF not found in response data: {e}")
    # Solution: Check TF name spelling, verify column names
```

## Performance Considerations

### Memory Management
- Large datasets loaded using chunked reading
- Unnecessary columns dropped early in processing
- Memory-efficient data types selected automatically

### I/O Optimization
- CSV reading optimized with appropriate engines
- Caching of preprocessed data for repeated access
- Lazy loading of optional data components

### Data Processing
- Vectorized operations for filtering and transformation
- Efficient indexing for data alignment
- Minimal data copying during preprocessing

## Related Classes

- **[BootstrappedModelingInputData](bootstrapped_input_data.md)**: Bootstrap sampling extension
- **[BootstrapModelResults](bootstrap_model_results.md)**: Results storage and aggregation
- **[StratifiedCV](stratified_cv.md)**: Cross-validation data handling

## Integration Points

The `ModelingInputData` class integrates with:

1. **CLI Interface**: Receives parameters from command-line arguments
2. **Bootstrap Modeling**: Provides base data for resampling
3. **Feature Engineering**: Supplies data for polynomial and interaction terms
4. **Cross-Validation**: Furnishes stratified sampling input
5. **Results Output**: Delivers metadata for result interpretation