# Input Data Formats

This guide provides detailed specifications for preparing input data for tfbpmodeling
analysis.

## Overview

tfbpmodeling requires two main input files plus optional supplementary files:

1. **Response File**: Gene expression data (dependent variable)
2. **Predictors File**: Transcription factor binding data (independent variables)
3. **Blacklist File** (optional): Features to exclude from analysis

## Response File Format

### Structure

The response file contains gene expression measurements:

```csv
gene_id,pTF1
YBR001C,0.234
YBR002W,-0.456
YBR003W,1.234
YBR004C,-0.123
```

### Requirements

| Element | Requirement | Description |
|---------|-------------|-------------|
| **Format** | CSV | Standard comma-separated values |
| **`gene_id` column** | Gene identifiers | Must match predictor file exactly |
| **Response column** | Named for the perturbed TF (e.g. `pTF1`) | Must match `--perturbed_tf`; numeric expression values |

## Predictors File Format

### Structure

The predictors file contains transcription factor binding data:

```csv
gene_id,TF_1,TF_2,TF_3,TF_4,pTF1
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
| **Data Cells** | Numeric values | Binding scores |
| **No Missing Values** | Complete data required | All cells must contain numeric values |

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