# Input Data Formats

This guide provides detailed specifications for preparing input data for tfbpmodeling analysis.

## Overview

tfbpmodeling requires two main input files plus optional supplementary files:

1. **Response File**: Gene expression data (dependent variable)
2. **Predictors File**: Transcription factor binding data (independent variables)
3. **Blacklist File** (optional): Features to exclude from analysis

## Response File Format

### Structure

The response file contains gene expression measurements:

```csv
gene_id,sample_1,sample_2,sample_3,sample_4,YPD1
YBR001C,0.234,-1.456,0.876,-0.123,0.543
YBR002W,-0.456,0.123,1.234,-0.567,-0.234
YBR003W,1.234,0.567,-0.234,0.876,0.123
YBR004C,-0.123,-0.876,0.456,0.234,-0.456
```

### Requirements

| Element | Requirement | Description |
|---------|-------------|-------------|
| **Format** | CSV with comma separators | Standard comma-separated values |
| **First Column** | Gene identifiers | Must match predictor file exactly |
| **Header Row** | Sample names + perturbed TF | Column names for each measurement |
| **Data Cells** | Numeric values | Expression measurements (log2 fold-change, z-scores, etc.) |
| **Perturbed TF Column** | Must be present | Column name must match `--perturbed_tf` parameter |

### Data Types

**Supported expression data types**:
- Log2 fold-change values
- Z-scores or standardized values
- Raw expression values (will be normalized if needed)
- Differential expression statistics

**Example expression data preparation**:

```python
import pandas as pd
import numpy as np

# Load raw expression data
raw_expr = pd.read_csv('raw_expression.csv', index_col=0)

# Calculate log2 fold-change vs control
control_samples = ['ctrl_1', 'ctrl_2', 'ctrl_3']
treatment_samples = ['treat_1', 'treat_2', 'treat_3']

control_mean = raw_expr[control_samples].mean(axis=1)
treatment_mean = raw_expr[treatment_samples].mean(axis=1)

log2fc = np.log2(treatment_mean + 1) - np.log2(control_mean + 1)

# Create response file
response_df = pd.DataFrame({
    'sample_1': log2fc,
    'sample_2': log2fc + np.random.normal(0, 0.1, len(log2fc)),
    'YPD1': log2fc  # Perturbed TF response
})

response_df.index.name = 'gene_id'
response_df.to_csv('response_data.csv')
```

## Predictors File Format

### Structure

The predictors file contains transcription factor binding data:

```csv
gene_id,TF_1,TF_2,TF_3,TF_4,YPD1_binding
YBR001C,0.123,0.456,0.789,0.012,0.345
YBR002W,0.234,0.567,0.890,0.123,0.456
YBR003W,0.345,0.678,0.901,0.234,0.567
YBR004C,0.456,0.789,0.012,0.345,0.678
```

### Requirements

| Element | Requirement | Description |
|---------|-------------|-------------|
| **Format** | CSV with comma separators | Standard comma-separated values |
| **First Column** | Gene identifiers | Must match response file exactly |
| **Header Row** | TF names | Transcription factor identifiers |
| **Data Cells** | Numeric values 0-1 | Binding probabilities or normalized scores |
| **No Missing Values** | Complete data required | All cells must contain numeric values |

### Data Types

**Supported binding data types**:
- ChIP-seq binding probabilities (0-1)
- Normalized binding scores (0-1)
- Peak presence indicators (0/1)
- Binding strength quantiles (0-1)

**Example binding data preparation**:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load ChIP-seq peak scores
chip_data = pd.read_csv('chip_peaks.csv', index_col=0)

# Normalize to 0-1 range
scaler = MinMaxScaler()
normalized_binding = pd.DataFrame(
    scaler.fit_transform(chip_data),
    index=chip_data.index,
    columns=chip_data.columns
)

# Save as predictors file
normalized_binding.index.name = 'gene_id'
normalized_binding.to_csv('binding_data.csv')
```

## Gene Identifier Consistency

### Critical Requirements

Both files **must** use identical gene identifiers:

```python
# Verify gene ID consistency
response_df = pd.read_csv('response.csv', index_col=0)
predictor_df = pd.read_csv('predictors.csv', index_col=0)

# Check for exact matches
common_genes = set(response_df.index) & set(predictor_df.index)
response_only = set(response_df.index) - set(predictor_df.index)
predictor_only = set(predictor_df.index) - set(response_df.index)

print(f"Common genes: {len(common_genes)}")
print(f"Response only: {len(response_only)}")
print(f"Predictor only: {len(predictor_only)}")
```

### Common Gene ID Formats

| Format | Example | Notes |
|--------|---------|-------|
| **Systematic names** | YBR001C | S. cerevisiae standard |
| **Gene symbols** | CDC42 | Human/mouse standard |
| **Ensembl IDs** | ENSG00000123456 | Cross-species compatible |
| **RefSeq IDs** | NM_001234567 | NCBI standard |

### Handling ID Mismatches

```python
# Align gene IDs between files
def align_gene_ids(response_df, predictor_df):
    # Find common genes
    common_genes = list(set(response_df.index) & set(predictor_df.index))

    # Subset both dataframes
    aligned_response = response_df.loc[common_genes]
    aligned_predictor = predictor_df.loc[common_genes]

    return aligned_response, aligned_predictor

# Apply alignment
aligned_response, aligned_predictor = align_gene_ids(response_df, predictor_df)

# Save aligned files
aligned_response.to_csv('aligned_response.csv')
aligned_predictor.to_csv('aligned_predictors.csv')
```

## Blacklist File Format

### Structure

Simple text file with one gene identifier per line:

```
YBR999W
YCR888X
ribosomal_protein_L1
housekeeping_gene_1
batch_effect_gene
```

### Use Cases

**Common genes to blacklist**:
- Housekeeping genes with stable expression
- Ribosomal protein genes
- Mitochondrial genes
- Known batch effect genes
- Genes with technical artifacts

### Creating Blacklist Files

```python
# Identify housekeeping genes
housekeeping_genes = [
    'ACT1', 'TUB1', 'TUB2', 'RDN18-1', 'RDN25-1'
]

# Identify low-variance genes
low_variance_genes = response_df.var(axis=1).sort_values().head(50).index.tolist()

# Combine blacklists
blacklist_genes = list(set(housekeeping_genes + low_variance_genes))

# Save blacklist
with open('blacklist.txt', 'w') as f:
    for gene in sorted(blacklist_genes):
        f.write(f"{gene}\n")
```

## Data Quality Checks

### Automated Validation

```python
def validate_input_data(response_file, predictor_file, perturbed_tf):
    """Validate input data format and consistency."""

    # Load data
    response_df = pd.read_csv(response_file, index_col=0)
    predictor_df = pd.read_csv(predictor_file, index_col=0)

    # Check 1: File formats
    assert response_df.index.name == 'gene_id', "Response file must have 'gene_id' as index"
    assert predictor_df.index.name == 'gene_id', "Predictor file must have 'gene_id' as index"

    # Check 2: Perturbed TF presence
    assert perturbed_tf in response_df.columns, f"Perturbed TF '{perturbed_tf}' not found in response"

    # Check 3: Gene ID overlap
    common_genes = set(response_df.index) & set(predictor_df.index)
    assert len(common_genes) > 100, f"Too few common genes: {len(common_genes)}"

    # Check 4: Data types
    assert response_df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
    assert predictor_df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()

    # Check 5: Missing values
    assert not response_df.isnull().any().any(), "Response data contains missing values"
    assert not predictor_df.isnull().any().any(), "Predictor data contains missing values"

    # Check 6: Value ranges
    predictor_ranges = predictor_df.describe()
    if (predictor_ranges.loc['min'] < 0).any():
        print("Warning: Some predictor values are negative")
    if (predictor_ranges.loc['max'] > 1).any():
        print("Warning: Some predictor values exceed 1")

    print("✓ All validation checks passed")

# Run validation
validate_input_data('response.csv', 'predictors.csv', 'YPD1')
```

### Manual Quality Assessment

```python
# Data exploration
def explore_data(response_file, predictor_file):
    response_df = pd.read_csv(response_file, index_col=0)
    predictor_df = pd.read_csv(predictor_file, index_col=0)

    print("=== Response Data ===")
    print(f"Shape: {response_df.shape}")
    print(f"Columns: {list(response_df.columns)}")
    print(f"Value range: [{response_df.min().min():.3f}, {response_df.max().max():.3f}]")
    print(f"Missing values: {response_df.isnull().sum().sum()}")

    print("\n=== Predictor Data ===")
    print(f"Shape: {predictor_df.shape}")
    print(f"Columns: {list(predictor_df.columns[:5])}..." if len(predictor_df.columns) > 5 else list(predictor_df.columns))
    print(f"Value range: [{predictor_df.min().min():.3f}, {predictor_df.max().max():.3f}]")
    print(f"Missing values: {predictor_df.isnull().sum().sum()}")

    # Plot distributions
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Response distribution
    response_df.iloc[:, 0].hist(bins=50, ax=axes[0])
    axes[0].set_title('Response Data Distribution')
    axes[0].set_xlabel('Expression Values')

    # Predictor distribution
    predictor_df.iloc[:, 0].hist(bins=50, ax=axes[1])
    axes[1].set_title('Predictor Data Distribution')
    axes[1].set_xlabel('Binding Values')

    plt.tight_layout()
    plt.savefig('data_distributions.png')
    plt.show()

# Explore your data
explore_data('response.csv', 'predictors.csv')
```

## Example Data Preparation Workflow

### Complete Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def prepare_tfbp_data(raw_expression_file, raw_binding_file,
                      perturbed_tf, output_prefix):
    """Complete data preparation pipeline."""

    # 1. Load raw data
    print("Loading raw data...")
    expr_raw = pd.read_csv(raw_expression_file, index_col=0)
    binding_raw = pd.read_csv(raw_binding_file, index_col=0)

    # 2. Process expression data
    print("Processing expression data...")
    # Log-transform if needed
    if expr_raw.min().min() >= 0:
        expr_processed = np.log2(expr_raw + 1)
    else:
        expr_processed = expr_raw

    # Add perturbed TF column (example: mean of treatment samples)
    if perturbed_tf not in expr_processed.columns:
        # Calculate as mean expression change
        expr_processed[perturbed_tf] = expr_processed.mean(axis=1)

    # 3. Process binding data
    print("Processing binding data...")
    # Normalize to 0-1 range
    scaler = MinMaxScaler()
    binding_processed = pd.DataFrame(
        scaler.fit_transform(binding_raw),
        index=binding_raw.index,
        columns=binding_raw.columns
    )

    # 4. Align gene IDs
    print("Aligning gene identifiers...")
    common_genes = list(set(expr_processed.index) & set(binding_processed.index))
    print(f"Common genes: {len(common_genes)}")

    expr_aligned = expr_processed.loc[common_genes]
    binding_aligned = binding_processed.loc[common_genes]

    # 5. Quality checks
    print("Running quality checks...")
    assert len(common_genes) > 100, "Insufficient gene overlap"
    assert perturbed_tf in expr_aligned.columns, "Perturbed TF missing"
    assert not expr_aligned.isnull().any().any(), "Missing expression data"
    assert not binding_aligned.isnull().any().any(), "Missing binding data"

    # 6. Save processed data
    print("Saving processed data...")
    expr_aligned.index.name = 'gene_id'
    binding_aligned.index.name = 'gene_id'

    expr_aligned.to_csv(f'{output_prefix}_response.csv')
    binding_aligned.to_csv(f'{output_prefix}_predictors.csv')

    print(f"✓ Data preparation complete!")
    print(f"  Response file: {output_prefix}_response.csv")
    print(f"  Predictors file: {output_prefix}_predictors.csv")
    print(f"  Genes: {len(common_genes)}")
    print(f"  Expression samples: {len(expr_aligned.columns)}")
    print(f"  TF predictors: {len(binding_aligned.columns)}")

# Run preparation
prepare_tfbp_data(
    raw_expression_file='raw_expression.csv',
    raw_binding_file='raw_binding.csv',
    perturbed_tf='YPD1',
    output_prefix='processed'
)
```

## Common Issues and Solutions

### Issue 1: Gene ID Mismatches

**Problem**: Gene IDs don't match between files

**Solution**: Use gene ID mapping:

```python
# Load ID mapping
id_mapping = pd.read_csv('gene_id_mapping.csv')  # old_id, new_id
mapping_dict = dict(zip(id_mapping['old_id'], id_mapping['new_id']))

# Apply mapping
response_df.index = response_df.index.map(mapping_dict).fillna(response_df.index)
```

### Issue 2: Missing Values

**Problem**: Missing data in binding matrix

**Solution**: Impute or filter:

```python
# Option 1: Remove genes with missing binding data
complete_genes = binding_df.dropna().index
response_df = response_df.loc[complete_genes]
binding_df = binding_df.loc[complete_genes]

# Option 2: Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
binding_imputed = pd.DataFrame(
    imputer.fit_transform(binding_df),
    index=binding_df.index,
    columns=binding_df.columns
)
```

### Issue 3: Scale Differences

**Problem**: Binding values not in 0-1 range

**Solution**: Normalize appropriately:

```python
# For ChIP-seq peak heights
binding_normalized = binding_df / binding_df.max()

# For count data
binding_normalized = (binding_df - binding_df.min()) / (binding_df.max() - binding_df.min())

# For already-processed scores
binding_clipped = binding_df.clip(0, 1)
```

This comprehensive guide should help you prepare properly formatted input data for tfbpmodeling analysis.