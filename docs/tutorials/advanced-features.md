# Advanced Features Tutorial

This tutorial covers advanced features and configuration options in tfbpmodeling for power users and specialized analyses.

## Overview

Beyond the basic workflow, tfbpmodeling offers advanced features for:

- **Feature Engineering**: Polynomial terms and custom variables
- **Model Tuning**: Advanced parameter optimization
- **Statistical Methods**: Alternative significance testing approaches
- **Performance Optimization**: High-performance computing configurations
- **Custom Workflows**: Programmatic usage and customization

## Feature Engineering

### Polynomial Terms

Add non-linear relationships to capture complex binding-expression dynamics:

```bash
# Add squared and cubic terms for the perturbed TF
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --squared_pTF \
    --cubic_pTF \
    --ptf_main_effect
```

**When to use polynomial terms**:
- Non-linear dose-response relationships
- Saturation effects in binding
- Threshold behaviors in gene expression

### Custom Variables

Include additional experimental variables in the model:

```bash
# Add batch effects and technical variables
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --add_model_variables "batch_id,plate_position,extraction_date" \
    --exclude_interactor_variables "batch_id,technical_replicate"
```

### Row-wise Features

Include summary statistics as predictors:

```bash
# Add maximum binding strength across all TFs
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --row_max \
    --normalize_sample_weights
```

## Advanced Model Configuration

### Iterative Dropout

Use iterative variable selection for feature-rich datasets:

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --iterative_dropout \
    --stabilization_ci_start 50.0 \
    --all_data_ci_level 95.0
```

**How iterative dropout works**:
1. Start with low confidence threshold (50%)
2. Remove non-significant features
3. Gradually increase threshold
4. Stabilize at final confidence level

### Stage 4 Configuration

Choose between linear and LassoCV approaches for final significance testing:

```bash
# Conservative LassoCV approach
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --stage4_lasso \
    --stage4_topn

# Sensitive linear regression approach (default)
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1
```

### Data Preprocessing

Advanced data handling options:

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --scale_by_std \
    --normalize_sample_weights \
    --bins "0,5,10,15,20,np.inf"
```

## High-Performance Computing

### Parallel Processing

Optimize for multi-core systems:

```bash
# Use all available cores
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --n_cpus 16 \
    --n_bootstraps 5000 \
    --max_iter 20000
```

### Memory Management

For large datasets:

```bash
# Reduce memory usage
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --n_bootstraps 500 \
    --top_n 300 \
    --max_iter 5000
```

### Cluster Computing

Example SLURM script for HPC environments:

```bash
#!/bin/bash
#SBATCH --job-name=tfbp_analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=24:00:00

module load python/3.11
source venv/bin/activate

python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file $SCRATCH/data/expression.csv \
    --predictors_file $SCRATCH/data/binding.csv \
    --perturbed_tf $1 \
    --n_cpus 32 \
    --n_bootstraps 10000 \
    --output_dir $SCRATCH/results \
    --log-handler file
```

## Programmatic Usage

### Python API

Use tfbpmodeling programmatically:

```python
import argparse
from tfbpmodeling.interface import linear_perturbation_binding_modeling

# Create arguments programmatically
args = argparse.Namespace(
    response_file='data/expression.csv',
    predictors_file='data/binding.csv',
    perturbed_tf='YPD1',
    n_bootstraps=2000,
    top_n=500,
    all_data_ci_level=95.0,
    topn_ci_level=85.0,
    max_iter=15000,
    iterative_dropout=True,
    stage4_lasso=True,
    squared_pTF=True,
    ptf_main_effect=True,
    output_dir='./results',
    output_suffix='_programmatic',
    n_cpus=8,
    random_state=42,
    # Set all other required parameters...
)

# Run analysis
linear_perturbation_binding_modeling(args)
```

### Batch Processing

Process multiple transcription factors:

```python
import os
from pathlib import Path

tfs_to_analyze = ['YPD1', 'YBR123W', 'YCR456X']
base_args = {
    'response_file': 'data/expression.csv',
    'predictors_file': 'data/binding.csv',
    'n_bootstraps': 2000,
    'random_state': 42,
    # ... other common parameters
}

for tf in tfs_to_analyze:
    print(f"Analyzing {tf}...")

    args = argparse.Namespace(
        perturbed_tf=tf,
        output_suffix=f'_batch_{tf}',
        **base_args
    )

    try:
        linear_perturbation_binding_modeling(args)
        print(f"✓ {tf} completed successfully")
    except Exception as e:
        print(f"✗ {tf} failed: {e}")
```

## Advanced Analysis Patterns

### Comparative Analysis

Compare different parameter settings:

```bash
# Conservative analysis
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --all_data_ci_level 99.0 \
    --topn_ci_level 95.0 \
    --stage4_lasso \
    --output_suffix _conservative \
    --random_state 42

# Sensitive analysis
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --all_data_ci_level 90.0 \
    --topn_ci_level 80.0 \
    --output_suffix _sensitive \
    --random_state 42
```

### Cross-Validation

Validate results across different data subsets:

```python
from sklearn.model_selection import KFold
import pandas as pd

# Load data
response_df = pd.read_csv('data/expression.csv', index_col=0)
predictor_df = pd.read_csv('data/binding.csv', index_col=0)

# Cross-validation across samples
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(response_df.columns[:-1])):
    # Create train/test splits
    train_samples = response_df.columns[train_idx]
    test_samples = response_df.columns[test_idx]

    train_response = response_df[list(train_samples) + ['YPD1']]
    test_response = response_df[list(test_samples) + ['YPD1']]

    # Save fold data
    train_response.to_csv(f'data/fold_{fold}_train_response.csv')
    test_response.to_csv(f'data/fold_{fold}_test_response.csv')

    # Run analysis on training data
    # ... (analysis code)
```

## Troubleshooting Advanced Features

### Convergence Issues

```bash
# Increase iterations and adjust tolerance
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --max_iter 50000 \
    --n_bootstraps 500  # Reduce for testing
```

### Memory Issues

```bash
# Reduce computational load
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --n_bootstraps 100 \
    --top_n 100 \
    --n_cpus 2
```

### Feature Selection Issues

```bash
# More lenient thresholds
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --all_data_ci_level 80.0 \
    --topn_ci_level 70.0
```

## Next Steps

- **[Input Formats](input-formats.md)**: Detailed data preparation
- **[CLI Reference](../cli/linear-perturbation-binding-modeling.md)**: Complete parameter documentation
- **[API Reference](../api/interface.md)**: Programmatic usage details