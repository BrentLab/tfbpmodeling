# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup
- `poetry install` - Install dependencies and set up development environment
- `pre-commit install` - Install pre-commit hooks for code quality checks

### Testing
- `poetry run pytest` - Run all tests
- `poetry run pytest --cov --cov-branch --cov-report=xml` - Run tests with coverage
- `poetry run pytest tfbpmodeling/tests/test_specific_module.py` - Run specific test file
- `poetry run pytest -k "test_function_name"` - Run specific test by name

### Code Quality
- `pre-commit run --all-files` - Run all pre-commit hooks manually
- `poetry run black .` - Format code with Black
- `poetry run flake8` - Check code style with Flake8
- `poetry run mypy tfbpmodeling` - Type checking with MyPy

### Documentation
- `mkdocs serve` - Start documentation server for live preview
- `mkdocs build` - Build documentation
- `poetry run mkdocs gh-deploy` - Deploy documentation to GitHub Pages

### Running the Application
- `poetry run python -m tfbpmodeling --help` - Show full help and all options

## Project Architecture

### Core Purpose
This package provides tools for transcription factor binding and perturbation (TFBP) modeling, specifically for analyzing relationships between transcription factor binding and gene expression perturbations using machine learning techniques.

### Main Components

#### Entry Point and Workflow (`__main__.py`)
All CLI argument definitions and the modeling workflow live in a single file. The
`main()` function builds the argparse parser and dispatches to `tfbpmodeling(args)`,
which runs the full sequential workflow:

- **Stage 0**: Preprocessing — input validation and output directory setup
- **Stage 1**: Bootstrap LassoCV on all data with the full interactor model; fits a
  best all-data model on the significant predictors
- **Stage 2**: Bootstrap LassoCV on the top-N data subset using Stage 1's significant
  predictors
- **Stage 3 - LassoCV Bootstrap** (optional, `--stage3_lassocv_bootstrap`): Refits surviving interactors
  with their independent main effects on all data using the Stage 1 protocol
- **Stage 3 - Lasso** (always runs): Tests significance of each surviving interactor
  term against its corresponding main effect

#### Key Modules
- `modeling_input_data.py` - Core data structures and preprocessing
- `bootstrapped_input_data.py` - Bootstrap resampling functionality
- `bootstrap_stratified_cv.py` - Cross-validation with stratification
- `bootstrap_stratified_cv_loop.py` - Iterative dropout variant of bootstrap CV
- `evaluate_interactor_significance_lassocv.py` - LassoCV-based significance testing
- `evaluate_interactor_significance_linear.py` - Linear regression significance testing
- `stratification_classification.py` - Data stratification logic
- `configure_logger.py` - Logger configuration utilities
- `utils/` - Utility functions for data manipulation

#### Data Flow
1. Input files: response data (gene expression) and predictors (binding data)
2. Data preprocessing with optional feature selection and binning
3. Bootstrap resampling with stratified cross-validation
4. Model fitting using LassoCV
5. Significance testing of surviving interaction terms against main effects
6. Output generation with confidence intervals and model statistics

### Development Patterns

#### Testing Structure
- Tests located in `tfbpmodeling/tests/`
- Each module has corresponding `test_*.py` file
- Use pytest with coverage reporting
- Tests ignore `tmp/` directory for exploratory work

#### Code Quality Standards
- Black code formatting (88 character line length)
- Type hints with MyPy checking
- Flake8 linting
- Pre-commit hooks enforce all quality checks
- Sphinx-style docstrings

#### Dependencies
- Scientific computing: numpy, scipy, pandas, scikit-learn
- Statistics: patsy for formula parsing
- Visualization: matplotlib, seaborn
- Development: pytest, black, mypy, mkdocs

### Important Configuration
- Python 3.11+ required
- Uses Poetry for dependency management
- Pre-commit hooks include security checks (detect-private-key)
- Coverage tracking excludes tests and experiments
- Pytest configuration ignores `tmp/` directory for local development
