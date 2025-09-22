# Basic Workflow Tutorial

This tutorial walks through a complete tfbpmodeling analysis from data preparation to result interpretation.

## Overview

We'll analyze the relationship between transcription factor binding and gene expression perturbation using a sample dataset. The workflow demonstrates:

1. **Data preparation**: Formatting input files
2. **Basic analysis**: Running with default parameters
3. **Result interpretation**: Understanding output files
4. **Parameter tuning**: Optimizing for your data

## Prerequisites

- tfbpmodeling installed and configured
- Basic familiarity with CSV files and command-line interfaces
- Understanding of transcription factor biology (helpful but not required)

## Sample Data

For this tutorial, we'll use example data representing:
- **Response data**: Gene expression changes after YPD1 knockout
- **Predictor data**: Transcription factor binding probabilities from ChIP-seq

### Creating Sample Data

```python
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create sample gene list
genes = [f"YBR{str(i).zfill(3)}W" for i in range(1, 1001)]
samples = [f"sample_{i}" for i in range(1, 101)]
tfs = [f"TF_{i}" for i in range(1, 51)]

# Generate response data (expression changes)
response_data = pd.DataFrame(
    np.random.normal(0, 1, (1000, 100)),
    index=genes,
    columns=samples
)
response_data.index.name = 'gene_id'

# Add YPD1 column (our perturbed TF)
response_data['YPD1'] = np.random.normal(-0.5, 0.8, 1000)

# Generate predictor data (binding probabilities)
predictor_data = pd.DataFrame(
    np.random.beta(0.5, 2, (1000, 50)),
    index=genes,
    columns=tfs
)
predictor_data.index.name = 'gene_id'

# Save to CSV
response_data.to_csv('tutorial_expression.csv')
predictor_data.to_csv('tutorial_binding.csv')

print("Sample data created:")
print(f"Response data: {response_data.shape}")
print(f"Predictor data: {predictor_data.shape}")
```

## Step 1: Basic Analysis

### Run Default Analysis

Start with the simplest possible command:

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file tutorial_expression.csv \
    --predictors_file tutorial_binding.csv \
    --perturbed_tf YPD1
```

This command will:
- Use 1000 bootstrap samples
- Apply 98% confidence interval for feature selection
- Process top 600 features in second round
- Save results to `./linear_perturbation_binding_modeling_results/YPD1_{timestamp}/`

### Monitor Progress

The command provides real-time progress information:

```
2024-01-15 14:30:22 - INFO - Starting linear perturbation binding modeling
2024-01-15 14:30:22 - INFO - Loading response data from: tutorial_expression.csv
2024-01-15 14:30:23 - INFO - Loading predictor data from: tutorial_binding.csv
2024-01-15 14:30:23 - INFO - Perturbed TF: YPD1
2024-01-15 14:30:23 - INFO - Data preprocessing complete
2024-01-15 14:30:23 - INFO - Features: 1000, Samples: 100
2024-01-15 14:30:24 - INFO - Starting Stage 1: Bootstrap modeling on all data
2024-01-15 14:30:24 - INFO - Bootstrap parameters: n_bootstraps=1000, random_state=None
2024-01-15 14:32:15 - INFO - Stage 1 complete. Significant features: 156
2024-01-15 14:32:15 - INFO - Starting Stage 2: Top-N modeling
2024-01-15 14:33:45 - INFO - Stage 2 complete. Refined features: 78
2024-01-15 14:33:45 - INFO - Starting Stage 3: Interactor significance testing
2024-01-15 14:34:20 - INFO - Analysis complete. Results saved to: ./linear_perturbation_binding_modeling_results/YPD1_20240115_143022/
```

## Step 2: Understanding Results

### Output Directory Structure

After completion, examine the results directory:

```bash
ls -la linear_perturbation_binding_modeling_results/YPD1_*/
```

```
YPD1_20240115_143022/
├── all_data_results/
│   ├── bootstrap_coefficients.csv
│   ├── confidence_intervals.csv
│   ├── model_statistics.csv
│   └── diagnostic_plots/
├── topn_results/
│   ├── bootstrap_coefficients.csv
│   ├── confidence_intervals.csv
│   ├── model_statistics.csv
│   └── diagnostic_plots/
├── interactor_significance/
│   ├── significance_results.csv
│   ├── comparison_statistics.csv
│   └── final_selection.csv
├── input_data/
│   ├── processed_response.csv
│   ├── processed_predictors.csv
│   └── data_summary.json
└── tfbpmodeling_20240115_143022.log
```

### Key Result Files

#### 1. Confidence Intervals (most important)

```bash
head -10 YPD1_*/all_data_results/confidence_intervals.csv
```

```csv
feature,mean_coef,std_coef,ci_lower,ci_upper,significant,abs_mean_coef
TF_1,0.023,0.008,0.007,0.039,True,0.023
TF_2,-0.045,0.012,-0.069,-0.021,True,0.045
TF_3,0.001,0.006,-0.011,0.013,False,0.001
```

**Key columns**:
- `feature`: Transcription factor name
- `mean_coef`: Average effect size across bootstrap samples
- `ci_lower/ci_upper`: Confidence interval bounds
- `significant`: Whether the effect is statistically significant

#### 2. Model Statistics

```bash
cat YPD1_*/all_data_results/model_statistics.csv
```

```csv
metric,value
mean_r2,0.234
std_r2,0.023
mean_cv_score,0.198
n_significant_features,156
total_features,1000
```

**Key metrics**:
- `mean_r2`: Model explanatory power
- `mean_cv_score`: Cross-validation performance
- `n_significant_features`: Count of statistically significant predictors

#### 3. Final Significant Interactions

```bash
head -10 YPD1_*/interactor_significance/final_selection.csv
```

```csv
feature,interaction_coef,main_effect_coef,p_value,significant,effect_size
TF_1:binding_strength,0.034,0.012,0.003,True,0.022
TF_2:binding_strength,-0.028,-0.008,0.012,True,0.020
```

This shows transcription factors with significant interaction effects beyond their main effects.

## Step 3: Interpreting Results

### Biological Interpretation

1. **Significant Features**: TFs with non-zero confidence intervals affect YPD1 expression
2. **Effect Direction**: Positive coefficients indicate binding increases expression
3. **Effect Size**: Larger absolute coefficients indicate stronger effects
4. **Interactions**: Features in final selection have context-dependent effects

### Statistical Interpretation

1. **Confidence Intervals**: 98% CIs that exclude zero are statistically significant
2. **Bootstrap Stability**: Lower standard deviations indicate more stable effects
3. **Cross-Validation**: CV scores show generalization performance
4. **Multiple Testing**: Built-in correction through bootstrap resampling

### Example Interpretation

From our results:
- **TF_1** (coef: 0.023): Binding increases YPD1 expression
- **TF_2** (coef: -0.045): Binding decreases YPD1 expression
- **TF_3** (coef: 0.001, not significant): No detectable effect

## Step 4: Parameter Optimization

### Increasing Statistical Power

For more robust results, increase bootstrap samples:

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file tutorial_expression.csv \
    --predictors_file tutorial_binding.csv \
    --perturbed_tf YPD1 \
    --n_bootstraps 2000 \
    --output_suffix _high_power
```

### Adjusting Sensitivity

For more sensitive detection, lower confidence thresholds:

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file tutorial_expression.csv \
    --predictors_file tutorial_binding.csv \
    --perturbed_tf YPD1 \
    --all_data_ci_level 95.0 \
    --topn_ci_level 85.0 \
    --output_suffix _sensitive
```

### Adding Feature Engineering

Include polynomial terms for non-linear relationships:

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file tutorial_expression.csv \
    --predictors_file tutorial_binding.csv \
    --perturbed_tf YPD1 \
    --squared_pTF \
    --ptf_main_effect \
    --row_max \
    --output_suffix _engineered
```

### Reproducible Analysis

For reproducible results, set random seed:

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file tutorial_expression.csv \
    --predictors_file tutorial_binding.csv \
    --perturbed_tf YPD1 \
    --random_state 42 \
    --output_suffix _reproducible
```

## Step 5: Comparing Results

### Compare Different Analyses

```bash
# List all result directories
ls -d YPD1_*/

# Compare significant feature counts
echo "Analysis,Significant_Features"
for dir in YPD1_*/; do
    count=$(tail -n +2 "$dir/all_data_results/confidence_intervals.csv" | awk -F',' '$6=="True"' | wc -l)
    echo "$dir,$count"
done
```

### Visualize Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load confidence intervals from different analyses
default_ci = pd.read_csv('YPD1_20240115_143022/all_data_results/confidence_intervals.csv')
sensitive_ci = pd.read_csv('YPD1_sensitive_20240115_143522/all_data_results/confidence_intervals.csv')

# Compare significant feature counts
print(f"Default analysis: {default_ci['significant'].sum()} significant features")
print(f"Sensitive analysis: {sensitive_ci['significant'].sum()} significant features")

# Plot effect size distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(default_ci['abs_mean_coef'], bins=30, alpha=0.7, label='Default')
ax1.set_xlabel('Absolute Effect Size')
ax1.set_ylabel('Frequency')
ax1.set_title('Default Analysis')

ax2.hist(sensitive_ci['abs_mean_coef'], bins=30, alpha=0.7, label='Sensitive', color='orange')
ax2.set_xlabel('Absolute Effect Size')
ax2.set_ylabel('Frequency')
ax2.set_title('Sensitive Analysis')

plt.tight_layout()
plt.savefig('effect_size_comparison.png')
plt.show()
```

## Next Steps

### For Your Own Data

1. **Prepare your files**: Follow the CSV format requirements
2. **Start simple**: Use default parameters first
3. **Validate results**: Check that results make biological sense
4. **Optimize parameters**: Adjust based on data characteristics
5. **Document analysis**: Save parameter choices and interpretations

### Advanced Techniques

- **[Advanced Features Tutorial](advanced-features.md)**: Feature engineering and model tuning
- **[Input Formats Guide](input-formats.md)**: Detailed data preparation instructions
- **[CLI Reference](../cli/linear-perturbation-binding-modeling.md)**: Complete parameter documentation

### Common Issues

#### Low R² Scores
- **Cause**: Weak signal, noisy data, or model misspecification
- **Solutions**: Increase sample size, add feature engineering, check data quality

#### Few Significant Features
- **Cause**: Stringent thresholds or weak effects
- **Solutions**: Lower confidence levels, increase bootstrap samples, check effect sizes

#### Long Runtime
- **Cause**: Large datasets or high bootstrap counts
- **Solutions**: Reduce parameters, increase `--n_cpus`, use smaller subset for testing

## Summary

This tutorial demonstrated:

1. **Basic analysis** with default parameters
2. **Result interpretation** using key output files
3. **Parameter optimization** for different analysis goals
4. **Comparison methods** for evaluating different approaches

The tfbpmodeling workflow provides a robust framework for analyzing transcription factor binding and perturbation relationships while controlling for multiple testing and providing interpretable results.