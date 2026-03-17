# tfbpmodeling

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![style](https://img.shields.io/badge/%20style-sphinx-0a507a.svg)](https://www.sphinx-doc.org/en/master/usage/index.html)
[![Pytest](https://github.com/BrentLab/tfbpmodeling/actions/workflows/ci.yml/badge.svg)](https://github.com/BrentLab/tfbpmodeling/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BrentLab/tfbpmodeling/graph/badge.svg?token=7zBsImRmjC)](https://codecov.io/gh/BrentLab/tfbpmodeling)

A Python package for **Transcription Factor Binding and Perturbation (TFBP) modeling** that analyzes relationships between transcription factor binding and gene expression perturbations using LASSO.

## What is tfbpmodeling?

tfbpmodeling provides a workflow for modeling the relationship between transcription factor binding data and gene expression perturbation data. The package uses bootstrap resampling and regularized regression (LassoCV) to identify significant binding-perturbation relationships while controlling for confounding factors.

### Workflow Overview

#### Stage 0 — Preprocessing

Data validation and preparation: load response and predictor files, apply blacklist, construct the model formula.

#### Stage 1 — All Data Modeling

Bootstrap LassoCV on the complete dataset using the full interactor model. Significant predictors are extracted at the specified confidence level (`--all_data_ci_level`). A best all-data model is then fit on those predictors.

#### Stage 2 — Top-N Modeling

Bootstrap LassoCV on the top-N data subset (genes ranked by perturbed TF binding) using the significant predictors from Stage 1. Surviving predictors are extracted at `--topn_ci_level`.

#### Stage 3 — Surviving interactor significance test

Tests each surviving interactor term against its corresponding main effect to determine whether the interaction provides genuine predictive value beyond the main effect alone.

##### Stage 3 - LassoCV Bootstrap (`--stage3_lassocv_bootstrap`, optional)

Before the significance test, refits the surviving interactors together with their independent main effects on all data using the same bootstrap LassoCV protocol as Stage 1. Use this when you want a regularized refit of the surviving terms before testing.

##### Stage 3 - Lasso

The significance test itself. For each surviving interactor, compares model fit with the interaction term against model fit with the corresponding main effect substituted in. Uses linear regression by default; pass `--stage3_lasso` to use LassoCV instead. Pass `--stage3_lasso_topn` to run the test on the top-N data subset rather than all data.

## Citation

If you use tfbpmodeling in your research, please cite:

```bibtex
@software{tfbpmodeling,
  title = {tfbpmodeling: Transcription Factor Binding and Perturbation Modeling},
  author = {Mateusiak, Chase and Erdenebaatar, Zolboo and Eric Jia and Mueller, Ben and Liu, Chenxing and Brent, Michael},
  url = {https://github.com/BrentLab/tfbpmodeling},
  year = {2025}
}
```

## Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/BrentLab/tfbpmodeling/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/BrentLab/tfbpmodeling/discussions)
- **Documentation**: Browse the complete documentation at [https://brentlab.github.io/tfbpmodeling/](https://brentlab.github.io/tfbpmodeling/)
