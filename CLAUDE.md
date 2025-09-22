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
- `poetry run python -m tfbpmodeling --help` - Show main help
- `poetry run python -m tfbpmodeling linear_perturbation_binding_modeling --help` - Show modeling command help

## Project Architecture

### Core Purpose
This package provides tools for transcription factor binding and perturbation (TFBP) modeling, specifically for analyzing relationships between transcription factor binding and gene expression perturbations using machine learning techniques.

### Main Components

#### Entry Point (`__main__.py`)
- Primary CLI interface using argparse
- Main command: `linear_perturbation_binding_modeling`
- Configurable logging with console/file handlers
- Extensible subcommand structure

#### Core Workflow (`interface.py`)
The main modeling workflow consists of 4 stages:
1. **Preprocessing**: Data validation and preparation
2. **Bootstrap Modeling**: LassoCV modeling with bootstrap resampling on all data
3. **Top-N Modeling**: Secondary modeling on significant predictors from top-performing data
4. **Interactor Significance**: Evaluation of interaction terms vs main effects

#### Key Modules
- `modeling_input_data.py` - Core data structures and preprocessing
- `bootstrapped_input_data.py` - Bootstrap resampling functionality
- `bootstrap_stratified_cv.py` - Cross-validation with stratification
- `evaluate_interactor_significance_*.py` - Statistical significance testing (LassoCV and linear methods)
- `stratification_classification.py` - Data stratification logic
- `utils/` - Utility functions for data manipulation

#### Data Flow
1. Input files: response data (gene expression) and predictors (binding data)
2. Data preprocessing with optional feature selection and binning
3. Bootstrap resampling with stratified cross-validation
4. Model fitting using LassoCV or linear regression
5. Significance testing of interaction terms
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
