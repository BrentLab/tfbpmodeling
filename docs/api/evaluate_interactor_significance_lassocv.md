# evaluate_interactor_significance_lassocv

LassoCV-based interactor significance testing for tfbpmodeling.

::: tfbpmodeling.evaluate_interactor_significance_lassocv

## Overview

The `evaluate_interactor_significance_lassocv` module provides functions for evaluating the significance of interaction terms using LassoCV regularization. This approach uses regularized regression to compare models with and without interaction terms, providing a conservative approach to interaction significance testing.

## Key Features

- **Regularized Comparison**: Uses LassoCV to compare interaction vs main effect models
- **Cross-Validation**: Built-in CV for robust model comparison
- **Conservative Testing**: Regularization reduces false positive interactions
- **Scalable Analysis**: Handles high-dimensional feature spaces efficiently

## Usage Examples

### Basic Significance Testing

```python
from tfbpmodeling.evaluate_interactor_significance_lassocv import (
    evaluate_interactor_significance_lassocv
)

# Run LassoCV-based significance testing
results = evaluate_interactor_significance_lassocv(
    X_main=main_effects_data,
    X_interaction=interaction_data,
    y=response_data,
    cv_folds=5,
    alpha_range=np.logspace(-4, 1, 50)
)

# Extract significant interactions
significant_interactions = results['significant_features']
p_values = results['p_values']
```

### Advanced Configuration

```python
# Custom LassoCV parameters
results = evaluate_interactor_significance_lassocv(
    X_main=main_effects_data,
    X_interaction=interaction_data,
    y=response_data,
    cv_folds=10,
    alpha_range=np.logspace(-5, 2, 100),
    max_iter=10000,
    tol=1e-6
)
```

## Method Details

### Statistical Approach

1. **Main Effect Model**: Fit LassoCV with only main effects
2. **Interaction Model**: Fit LassoCV with main effects + interactions
3. **Model Comparison**: Compare CV scores and coefficient stability
4. **Significance Assessment**: Determine if interactions improve model performance

### Advantages

- **Regularization**: Reduces overfitting in high-dimensional settings
- **Feature Selection**: Automatically selects relevant interactions
- **Robust**: Less sensitive to noise compared to standard linear regression
- **Scalable**: Efficient for large feature sets

### Considerations

- **Conservative**: May miss weak but real interactions
- **Hyperparameter Sensitive**: Alpha range affects results
- **Interpretation**: Regularized coefficients may be shrunk

## Related Modules

- **[evaluate_interactor_significance_linear](evaluate_interactor_significance_linear.md)**: Linear regression alternative
- **[interactor_significance_results](interactor_significance_results.md)**: Results handling
- **[interface](interface.md)**: Workflow integration