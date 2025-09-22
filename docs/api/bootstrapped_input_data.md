# bootstrapped_input_data

Bootstrap resampling functionality for tfbpmodeling input data.

::: tfbpmodeling.bootstrapped_input_data

## Overview

The `bootstrapped_input_data` module provides the `BootstrappedModelingInputData` class that extends the base `ModelingInputData` with bootstrap resampling capabilities. This is essential for the statistical inference approach used in tfbpmodeling.

## Key Features

- **Bootstrap Sample Generation**: Creates multiple resampled datasets from the original data
- **Stratified Sampling**: Maintains data distribution characteristics across bootstrap samples
- **Reproducible Results**: Supports random seed setting for consistent results
- **Memory Efficient**: Optimized storage and access patterns for large bootstrap sets

## Usage Examples

### Basic Bootstrap Creation

```python
from tfbpmodeling.modeling_input_data import ModelingInputData
from tfbpmodeling.bootstrapped_input_data import BootstrappedModelingInputData

# Create base data
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

### Accessing Bootstrap Samples

```python
# Get bootstrap indices
indices = bootstrap_data.get_bootstrap_indices()

# Get specific bootstrap sample data
sample_data = bootstrap_data.get_bootstrap_sample(sample_idx=0)

# Iterate through all bootstrap samples
for i in range(bootstrap_data.n_bootstraps):
    sample = bootstrap_data.get_bootstrap_sample(i)
    # Process sample...
```

## Related Modules

- **[modeling_input_data](modeling_input_data.md)**: Base data structures
- **[bootstrap_model_results](bootstrap_model_results.md)**: Results aggregation
- **[interface](interface.md)**: Main workflow integration