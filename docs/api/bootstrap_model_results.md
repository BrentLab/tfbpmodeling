# bootstrap_model_results

Results aggregation and statistical analysis for bootstrap modeling.

::: tfbpmodeling.bootstrap_model_results

## Overview

The `bootstrap_model_results` module provides classes and functions for aggregating and analyzing results from bootstrap modeling. It handles the statistical analysis of coefficient distributions, confidence intervals, and significance testing.

## Key Features

- **Coefficient Aggregation**: Combines results from multiple bootstrap samples
- **Confidence Interval Calculation**: Computes percentile-based confidence intervals
- **Statistical Significance**: Determines feature significance based on CI bounds
- **Result Export**: Saves results in multiple formats for analysis

## Usage Examples

### Result Aggregation

```python
from tfbpmodeling.bootstrap_model_results import BootstrapModelResults

# Create results aggregator
results = BootstrapModelResults(
    bootstrap_coefficients=coef_matrix,
    feature_names=feature_list,
    confidence_level=95.0
)

# Get confidence intervals
ci_results = results.get_confidence_intervals()

# Get significant features
significant_features = results.get_significant_features()
```

### Statistical Analysis

```python
# Calculate summary statistics
summary_stats = results.get_summary_statistics()

# Export results
results.save_results(output_dir='./results/')

# Generate diagnostic plots
results.plot_coefficient_distributions()
```

## Related Modules

- **[bootstrapped_input_data](bootstrapped_input_data.md)**: Bootstrap data generation
- **[interface](interface.md)**: Main workflow integration
- **[interactor_significance_results](interactor_significance_results.md)**: Interaction analysis