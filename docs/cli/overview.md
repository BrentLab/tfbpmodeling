# CLI Reference Overview

tfbpmodeling provides a comprehensive command-line interface for transcription factor binding and perturbation modeling. The CLI is designed to be user-friendly while offering advanced options for power users.

## Main Command Structure

```bash
python -m tfbpmodeling [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

### Global Options

Options that apply to all commands:

| Option | Description | Default | Choices |
|--------|-------------|---------|---------|
| `--log-level` | Set logging verbosity | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--log-handler` | Log output destination | `console` | `console`, `file` |

### Available Commands

Currently, tfbpmodeling provides one main command:

- **[`linear_perturbation_binding_modeling`](linear-perturbation-binding-modeling.md)**: Complete workflow for TFBP analysis

## Command Help System

### Getting Help

Display main help:
```bash
python -m tfbpmodeling --help
```

Display command-specific help:
```bash
python -m tfbpmodeling linear_perturbation_binding_modeling --help
```

### Help Output Format

The CLI uses a custom help formatter that organizes options into logical groups:

- **Input**: Data files and basic parameters
- **Feature Options**: Feature engineering and selection
- **Binning Options**: Data stratification parameters
- **Parameters**: Model configuration and thresholds
- **Output**: Result directories and naming
- **System**: Performance and logging options

## Common Usage Patterns

### Basic Analysis
```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data.csv \
    --predictors_file binding.csv \
    --perturbed_tf YPD1
```

### Reproducible Analysis
```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data.csv \
    --predictors_file binding.csv \
    --perturbed_tf YPD1 \
    --random_state 42 \
    --log-level DEBUG \
    --log-handler file
```

### High-Performance Analysis
```bash
python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file data.csv \
    --predictors_file binding.csv \
    --perturbed_tf YPD1 \
    --n_cpus 16 \
    --n_bootstraps 5000
```

## Exit Codes

The CLI uses standard exit codes:

- **0**: Success
- **1**: General error (invalid arguments, file not found, etc.)
- **2**: Modeling error (convergence failure, insufficient data, etc.)

## Logging

### Log Levels

| Level | Description | When to Use |
|-------|-------------|-------------|
| `DEBUG` | Detailed diagnostic information | Development, troubleshooting |
| `INFO` | General information about progress | Normal operation |
| `WARNING` | Warning messages about potential issues | Production monitoring |
| `ERROR` | Error messages for recoverable problems | Error investigation |
| `CRITICAL` | Critical errors that stop execution | System failures |

### Log Handlers

#### Console Handler (default)
Outputs log messages to the terminal with color coding:
```bash
--log-handler console
```

#### File Handler
Saves log messages to a timestamped file:
```bash
--log-handler file
```

Creates log files named: `tfbpmodeling_YYYYMMDD-HHMMSS.log`

### Example Log Output

```
2024-01-15 14:30:22,123 - INFO - Starting linear perturbation binding modeling
2024-01-15 14:30:22,125 - INFO - Loading response data from: data/expression.csv
2024-01-15 14:30:22,234 - INFO - Loading predictor data from: data/binding.csv
2024-01-15 14:30:22,456 - INFO - Perturbed TF: YPD1
2024-01-15 14:30:22,458 - INFO - Starting Stage 1: Bootstrap modeling on all data
2024-01-15 14:30:22,459 - DEBUG - Bootstrap parameters: n_bootstraps=1000, random_state=None
```

## Configuration Files

While tfbpmodeling doesn't currently support configuration files, you can create shell scripts or aliases for commonly used parameter combinations:

### Shell Script Example

```bash
#!/bin/bash
# run_analysis.sh

RESPONSE_FILE="$1"
PREDICTORS_FILE="$2"
PERTURBED_TF="$3"
OUTPUT_DIR="${4:-./results}"

python -m tfbpmodeling linear_perturbation_binding_modeling \
    --response_file "$RESPONSE_FILE" \
    --predictors_file "$PREDICTORS_FILE" \
    --perturbed_tf "$PERTURBED_TF" \
    --output_dir "$OUTPUT_DIR" \
    --n_bootstraps 2000 \
    --squared_pTF \
    --ptf_main_effect \
    --iterative_dropout \
    --random_state 42 \
    --log-level INFO \
    --log-handler file
```

Usage:
```bash
./run_analysis.sh expression.csv binding.csv YPD1 my_results
```

### Bash Alias Example

```bash
# Add to ~/.bashrc or ~/.bash_profile
alias tfbp-basic='python -m tfbpmodeling linear_perturbation_binding_modeling'
alias tfbp-advanced='python -m tfbpmodeling linear_perturbation_binding_modeling --n_bootstraps 2000 --squared_pTF --ptf_main_effect --iterative_dropout --random_state 42'
```

Usage:
```bash
tfbp-basic --response_file data.csv --predictors_file binding.csv --perturbed_tf YPD1
```

## Error Handling

### Common Error Messages

#### File Not Found
```
ERROR: Response file not found: data/missing_file.csv
```
**Solution**: Verify file paths and permissions

#### Invalid Perturbed TF
```
ERROR: Perturbed TF 'INVALID_TF' not found in response file columns
```
**Solution**: Check TF name spelling and presence in response file

#### Insufficient Data
```
ERROR: Insufficient data after filtering. Found 5 samples, minimum required: 10
```
**Solution**: Check data quality, reduce filtering, or provide more samples

#### Convergence Issues
```
WARNING: LassoCV failed to converge for 15/1000 bootstrap samples
```
**Solution**: Increase `--max_iter` or check data preprocessing

### Debugging Tips

1. **Start with defaults**: Use minimal parameters first
2. **Enable debug logging**: Add `--log-level DEBUG`
3. **Use file logging**: Add `--log-handler file` to preserve logs
4. **Check input data**: Verify file formats and content
5. **Reduce complexity**: Lower bootstrap samples for initial testing

## Performance Considerations

### Memory Usage
- Memory usage scales with: number of features × number of samples × bootstrap samples
- For large datasets, consider reducing `--n_bootstraps` or `--top_n`

### CPU Usage
- Set `--n_cpus` to match your system capabilities
- Each LassoCV call uses the specified number of CPUs
- Default of 4 CPUs works well for most systems

### Runtime Estimation
Approximate runtime factors:
- **Bootstrap samples**: Linear scaling
- **Feature count**: Quadratic scaling with regularization
- **Sample count**: Linear scaling
- **CPU cores**: Near-linear speedup

For typical datasets (1000 features, 100 samples):
- 1000 bootstraps: ~10-30 minutes
- 2000 bootstraps: ~20-60 minutes
- 5000 bootstraps: ~1-3 hours

## Next Steps

- **[Linear Perturbation Binding Modeling](linear-perturbation-binding-modeling.md)**: Detailed documentation for the main command
- **[Tutorials](../tutorials/basic-workflow.md)**: Step-by-step examples
- **[API Reference](../api/interface.md)**: Programmatic usage documentation