# evaluate_interactor_significance_linear

Linear regression-based interactor significance testing for tfbpmodeling.

::: tfbpmodeling.evaluate_interactor_significance_linear

## Overview

The `evaluate_interactor_significance_linear` module provides functions for evaluating the significance of interaction terms using standard linear regression methods. This approach uses classical statistical tests to compare models with and without interaction terms.

## Key Features

- **Classical Statistics**: Uses standard linear regression and F-tests
- **Direct Interpretation**: Unregularized coefficients with clear interpretation
- **Statistical Rigor**: Proper p-values and confidence intervals
- **Flexible Testing**: Multiple comparison correction options

## Usage Examples

### Basic Significance Testing

```python
from tfbpmodeling.evaluate_interactor_significance_linear import (
    evaluate_interactor_significance_linear
)

# Run linear regression-based significance testing
results = evaluate_interactor_significance_linear(
    X_main=main_effects_data,
    X_interaction=interaction_data,
    y=response_data,
    alpha=0.05
)

# Extract results
significant_interactions = results['significant_features']
p_values = results['p_values']
f_statistics = results['f_statistics']
```

### Multiple Comparison Correction

```python
# With Bonferroni correction
results = evaluate_interactor_significance_linear(
    X_main=main_effects_data,
    X_interaction=interaction_data,
    y=response_data,
    alpha=0.05,
    correction='bonferroni'
)

# With FDR correction
results = evaluate_interactor_significance_linear(
    X_main=main_effects_data,
    X_interaction=interaction_data,
    y=response_data,
    alpha=0.05,
    correction='fdr_bh'
)
```

## Method Details

### Statistical Approach

1. **Main Effect Model**: Fit linear regression with only main effects
2. **Full Model**: Fit linear regression with main effects + interactions
3. **F-Test**: Compare models using F-statistic for nested model comparison
4. **Individual Tests**: Test each interaction term individually

### Model Comparison

- **Nested F-Test**: Overall test for any interaction effects
- **Individual t-Tests**: Test each interaction coefficient
- **Partial F-Tests**: Test subsets of interaction terms
- **Multiple Comparisons**: Adjust for multiple testing

### Advantages

- **Interpretable**: Direct coefficient interpretation
- **Established Theory**: Well-understood statistical properties
- **Sensitive**: Can detect small but significant effects
- **Comprehensive**: Provides full statistical inference

### Considerations

- **Overfitting Risk**: May overfit in high-dimensional settings
- **Multicollinearity**: Sensitive to correlated predictors
- **Assumptions**: Requires standard linear regression assumptions
- **Multiple Testing**: Needs correction for many interactions

## Related Modules

- **[evaluate_interactor_significance_lassocv](evaluate_interactor_significance_lassocv.md)**: Regularized alternative
- **[interactor_significance_results](interactor_significance_results.md)**: Results handling
- **[interface](interface.md)**: Workflow integration