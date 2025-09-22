# stratified_cv_r2

R² calculation with stratified cross-validation for tfbpmodeling.

::: tfbpmodeling.stratified_cv_r2

## Overview

The `stratified_cv_r2` module provides specialized functions for calculating R² scores using stratified cross-validation. This ensures that model performance metrics accurately reflect the model's ability to generalize across different data strata.

## Key Features

- **Stratified R² Calculation**: R² scores that account for data stratification
- **Cross-Validation Integration**: Works with stratified CV folds
- **Bootstrap Compatibility**: Integrates with bootstrap resampling
- **Robust Performance Metrics**: Reduces bias in performance estimation

## Usage Examples

### Basic R² Calculation

```python
from tfbpmodeling.stratified_cv_r2 import calculate_stratified_r2

# Calculate stratified R² scores
r2_scores = calculate_stratified_r2(
    estimator=LassoCV(),
    X=predictor_data,
    y=response_data,
    cv_folds=5,
    stratification_bins=[0, 8, 12, np.inf]
)

print(f"Mean R²: {r2_scores.mean():.3f}")
print(f"Std R²: {r2_scores.std():.3f}")
```

### Bootstrap Integration

```python
from tfbpmodeling.stratified_cv_r2 import bootstrap_stratified_r2

# Bootstrap R² with stratification
bootstrap_r2 = bootstrap_stratified_r2(
    estimator=LassoCV(),
    X=predictor_data,
    y=response_data,
    n_bootstraps=1000,
    cv_folds=5,
    stratification_bins=[0, 8, 12, np.inf]
)

# Get confidence interval for R²
r2_ci = np.percentile(bootstrap_r2, [2.5, 97.5])
print(f"R² 95% CI: [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")
```

## Performance Metrics

### Stratified R²

Calculates R² separately for each stratum and then aggregates:

```python
# Per-stratum R² calculation
stratum_r2 = calculate_per_stratum_r2(
    estimator=model,
    X=X_test,
    y=y_test,
    strata=test_strata
)
```

### Weighted Aggregation

Combines R² scores across strata with appropriate weighting:

```python
# Weighted average R²
weighted_r2 = calculate_weighted_r2(
    stratum_r2_scores=stratum_scores,
    stratum_weights=stratum_sizes
)
```

## Related Modules

- **[stratified_cv](stratified_cv.md)**: Stratified cross-validation
- **[bootstrap_model_results](bootstrap_model_results.md)**: Results aggregation
- **[interface](interface.md)**: Workflow integration