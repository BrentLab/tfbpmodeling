# interactor_significance_results

Results and analysis for transcription factor interaction significance testing.

::: tfbpmodeling.interactor_significance_results

## Overview

The `interactor_significance_results` module provides classes for storing, analyzing, and reporting the results of interaction significance testing. This is used in Stage 4 of the tfbpmodeling workflow to evaluate whether interaction terms provide significant explanatory power beyond main effects.

## Key Features

- **Interaction vs Main Effect Comparison**: Statistical comparison of interaction and main effect models
- **Significance Testing**: P-value calculation and hypothesis testing
- **Effect Size Analysis**: Quantification of interaction effect magnitudes
- **Result Summarization**: Comprehensive reporting of significant interactions

## Usage Examples

### Basic Usage

```python
from tfbpmodeling.interactor_significance_results import InteractorSignificanceResults

# Create results object
results = InteractorSignificanceResults(
    interaction_effects=interaction_coefs,
    main_effects=main_coefs,
    p_values=p_vals,
    feature_names=features
)

# Get significant interactions
significant_interactions = results.get_significant_interactions(alpha=0.05)

# Export results
results.save_results('./output/')
```

### Analysis and Reporting

```python
# Generate summary statistics
summary = results.get_summary_statistics()

# Plot interaction effects
results.plot_interaction_effects()

# Create comparison table
comparison_table = results.create_comparison_table()
```

## Related Modules

- **[evaluate_interactor_significance_lassocv](evaluate_interactor_significance_lassocv.md)**: LassoCV-based testing
- **[evaluate_interactor_significance_linear](evaluate_interactor_significance_linear.md)**: Linear regression-based testing
- **[bootstrap_model_results](bootstrap_model_results.md)**: Bootstrap result aggregation