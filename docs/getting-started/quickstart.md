# Quick Start

## Installation

```bash
pip install git+https://github.com/BrentLab/tfbpmodeling.git
```

## Create Example Data

The script below generates synthetic but structurally valid input files so you
can run the full workflow without real experimental data.

```python
import pandas as pd
import numpy as np

np.random.seed(42)

genes = [f"YBR{str(i).zfill(3)}W" for i in range(1, 1001)]
tfs = [f"TF_{i}" for i in range(1, 51)]

# Response: one column named for the perturbed TF
response_data = pd.DataFrame(
    {"pTF1": np.random.normal(-0.5, 0.8, 1000)},
    index=genes,
)
response_data.index.name = "gene_id"
response_data.to_csv("tutorial_expression.csv")

# Predictors: binding scores; the perturbed TF must be present as a column
predictor_data = pd.DataFrame(
    np.random.beta(0.5, 2, (1000, 50)),
    index=genes,
    columns=tfs,
)
predictor_data.index.name = "gene_id"
predictor_data["pTF1"] = np.random.beta(0.5, 2, 1000)
predictor_data.to_csv("tutorial_binding.csv")
```

See [Input Data Formats](input-formats.md) for the full format specification.

## Run the Analysis

```bash
python -m tfbpmodeling \
    --response_file tutorial_expression.csv \
    --predictors_file tutorial_binding.csv \
    --perturbed_tf pTF1 \
    --random_state 42
```

Pass `--random_state` whenever you need reproducible results. Run
`python -m tfbpmodeling --help` for the full list of options.

## Examine Results

Results are written to `./tfbpmodeling_results/pTF1/` by default. The key files:

- `all_data_significant_{ci}.json` — predictors surviving Stage 1
- `topn_significant_{ci}.json` — predictors surviving Stage 2
- `stage3_lassocv_significance_results.json` — interactor significance results

See [Output Reference](../output.md) for a complete description of every file.

## Quick Visualization

```python
import json
import matplotlib.pyplot as plt

with open("tfbpmodeling_results/pTF1/all_data_significant_98-0.json") as f:
    sig = json.load(f)

features = list(sig.keys())
coefs = [sig[f] for f in features]

plt.figure(figsize=(10, 4))
plt.bar(range(len(features)), coefs)
plt.xticks(range(len(features)), features, rotation=90, fontsize=7)
plt.ylabel("Coefficient")
plt.title("Significant predictors — Stage 1")
plt.tight_layout()
plt.savefig("significant_predictors.png")
plt.show()
```

## Next Steps

- [Input Data Formats](input-formats.md) — format specification for real data
- [Output Reference](../output.md) — full description of output files
- [Cluster Usage](../tutorials/cluster-usage.md) — running at scale with SLURM
