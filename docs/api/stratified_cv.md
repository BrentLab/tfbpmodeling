# stratified_cv

Stratified cross-validation for tfbpmodeling.

::: tfbpmodeling.stratified_cv

## Overview

The `stratified_cv` module provides cross-validation functionality that maintains the distribution of data characteristics across folds. This is particularly important for tfbpmodeling where data may have natural groupings or strata that should be preserved during validation.

## Key Features

- **Stratified Sampling**: Maintains data distribution across CV folds
- **Bootstrap Integration**: Works with bootstrap resampling
- **Flexible Stratification**: Multiple stratification strategies
- **Robust Validation**: Reduces bias in cross-validation estimates

## Usage Examples

### Basic Stratified CV

```python
from tfbpmodeling.stratified_cv import StratifiedCV

# Create stratified CV object
cv = StratifiedCV(
    n_splits=5,
    stratification_variable='binding_strength_bins',
    random_state=42
)

# Generate CV folds
for train_idx, test_idx in cv.split(X, y):
    # Train and evaluate model
    pass
```

### Bootstrap Integration

```python
from tfbpmodeling.stratified_cv import bootstrap_stratified_cv

# Perform bootstrap with stratified CV
cv_scores = bootstrap_stratified_cv(
    X=predictor_data,
    y=response_data,
    estimator=LassoCV(),
    n_bootstraps=1000,
    cv_folds=5,
    stratification_bins=[0, 8, 12, np.inf]
)
```

## Stratification Methods

### Binding Strength Bins

Stratifies data based on transcription factor binding strength:

```python
# Define binding strength bins
bins = [0, 0.1, 0.5, 1.0]  # Low, medium, high binding

cv = StratifiedCV(
    n_splits=5,
    stratification_method='binding_bins',
    bins=bins
)
```

### Expression Level Bins

Stratifies based on expression level ranges:

```python
# Expression-based stratification
cv = StratifiedCV(
    n_splits=5,
    stratification_method='expression_bins',
    bins=[-np.inf, -1, 0, 1, np.inf]
)
```

## Related Modules

- **[stratified_cv_r2](stratified_cv_r2.md)**: R² calculation with stratification
- **[bootstrapped_input_data](bootstrapped_input_data.md)**: Bootstrap data handling
- **[interface](interface.md)**: Workflow integration