# Quick Start Guide

This guide will walk you through your first analysis with tfbpmodeling using example data.

## Overview

tfbpmodeling analyzes the relationship between transcription factor binding data and gene expression perturbation data through a 4-stage workflow:

1. **Bootstrap modeling** on all data to identify significant binding-expression relationships
2. **Top-N modeling** on the most significant predictors from high-performing data
3. **Interaction analysis** to evaluate interaction terms vs main effects
4. **Results generation** with comprehensive statistics and confidence intervals

## Prepare Your Data

tfbpmodeling requires two main input files:

### Response File (Gene Expression Data)

CSV format with genes as rows and samples as columns:

```csv
gene_id,sample1,sample2,sample3,sample4
YPD1,0.23,-1.45,0.87,-0.12
YBR123W,1.34,0.56,-0.23,0.78
YCR456X,-0.45,0.12,1.23,-0.56
```

- First column: Gene identifiers
- Subsequent columns: Expression values for each sample
- Must contain a column matching your `--perturbed_tf` parameter

### Predictors File (Binding Data)

CSV format with genes as rows and transcription factors as columns:

```csv
gene_id,TF1,TF2,TF3,TF4
YPD1,0.34,0.12,0.78,0.01
YBR123W,0.89,0.45,0.23,0.67
YCR456X,0.12,0.78,0.34,0.90
```

- First column: Gene identifiers (must match response file)
- Subsequent columns: Binding measurements for different TFs

## Basic Analysis

### Minimal Command

Run a basic analysis with default parameters:

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1
```

This command will:

- Use 1000 bootstrap samples
- Apply 98% confidence interval for initial feature selection
- Use 90% confidence interval for second-round modeling
- Process top 600 features in the second round
- Save results to `./linear_perturbation_binding_modeling_results/YPD1/`

### With Custom Parameters

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --n_bootstraps 2000 \
    --top_n 500 \
    --all_data_ci_level 95.0 \
    --topn_ci_level 85.0 \
    --output_dir ./my_results \
    --output_suffix _custom_analysis \
    --random_state 42
```

## Understanding the Output

Results are saved in a timestamped subdirectory within your specified output directory:

```
my_results/YPD1_custom_analysis_20240115_143022/
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
└── tfbpmodeling_20240115_143022.log
```

### Key Output Files

#### Bootstrap Coefficients
Contains coefficient estimates from each bootstrap sample:
- Rows: Bootstrap samples
- Columns: Model features
- Values: Coefficient estimates

#### Confidence Intervals
Statistical significance of each feature:
- `feature`: Feature name
- `mean_coef`: Mean coefficient across bootstrap samples
- `ci_lower`: Lower confidence interval bound
- `ci_upper`: Upper confidence interval bound
- `significant`: Boolean indicating statistical significance

#### Model Statistics
Overall model performance metrics:
- R² scores across bootstrap samples
- Cross-validation performance
- Feature selection statistics

## Advanced Features

### Feature Engineering

Add polynomial terms and custom variables:

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --squared_pTF \
    --cubic_pTF \
    --row_max \
    --ptf_main_effect \
    --add_model_variables "red_median,green_median"
```

### Data Processing Options

Control data preprocessing:

```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --normalize_sample_weights \
    --scale_by_std \
    --bins "0,5,10,15,np.inf"
```

### Excluding Features

Exclude specific genes or features:

```bash
# Create blacklist file
echo -e "YBR999W\nYCR888X\ncontrol_gene" > blacklist.txt

# Run analysis with exclusions
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data/expression.csv \
    --predictors_file data/binding.csv \
    --perturbed_tf YPD1 \
    --blacklist_file blacklist.txt \
    --exclude_interactor_variables "batch_effect,technical_replicate"
```

## Example Workflow

Here's a complete example workflow:

### 1. Prepare Data
```bash
# Create example data directory
mkdir -p example_data

# Your data preparation steps here
# (load and format your actual expression and binding data)
```

### 2. Run Basic Analysis
```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file example_data/expression.csv \
    --predictors_file example_data/binding.csv \
    --perturbed_tf YPD1 \
    --random_state 12345 \
    --output_dir ./results \
    --output_suffix _basic_analysis
```

### 3. Run Advanced Analysis
```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file example_data/expression.csv \
    --predictors_file example_data/binding.csv \
    --perturbed_tf YPD1 \
    --n_bootstraps 2000 \
    --squared_pTF \
    --ptf_main_effect \
    --iterative_dropout \
    --stage4_lasso \
    --random_state 12345 \
    --output_dir ./results \
    --output_suffix _advanced_analysis
```

### 4. Compare Results
```bash
# Compare the two analyses
ls -la results/YPD1_*_analysis_*/
```

## Next Steps

- **[CLI Reference](../cli/overview.md)**: Complete documentation of all command-line options
- **[Tutorials](../tutorials/basic-workflow.md)**: Detailed tutorials with real examples
- **[API Reference](../api/interface.md)**: Documentation for programmatic usage
- **[Input Formats](../tutorials/input-formats.md)**: Detailed specifications for input data

## Troubleshooting

### Common Issues

#### File Not Found
```bash
# Verify your files exist and paths are correct
ls -la data/expression.csv data/binding.csv
```

#### Memory Issues
```bash
# Reduce bootstrap samples or top_n for large datasets
--n_bootstraps 500 --top_n 300
```

#### Convergence Issues
```bash
# Increase iteration limit
--max_iter 20000
```

#### No Significant Features
```bash
# Lower confidence intervals or check data quality
--all_data_ci_level 90.0 --topn_ci_level 80.0
```

For more help, see the [troubleshooting section](../development/testing.md) or open an issue on GitHub.