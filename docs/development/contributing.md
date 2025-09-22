# Contributing to tfbpmodeling

We welcome contributions to tfbpmodeling! This guide will help you get started with contributing code, documentation, or bug reports.

## Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub
   git clone https://github.com/YOUR_USERNAME/tfbpmodeling.git
   cd tfbpmodeling

   # Add upstream remote
   git remote add upstream https://github.com/BrentLab/tfbpmodeling.git
   ```

2. **Install Dependencies**
   ```bash
   # Install Poetry if you haven't already
   pip install poetry

   # Configure Poetry (recommended)
   poetry config virtualenvs.in-project true

   # Install all dependencies including dev tools
   poetry install
   ```

3. **Set Up Pre-commit Hooks**
   ```bash
   poetry run pre-commit install
   ```

### Development Workflow

1. **Create Feature Branch**
   ```bash
   # Start from dev branch
   git checkout dev
   git pull upstream dev

   # Create feature branch
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run tests
   poetry run pytest

   # Run with coverage
   poetry run pytest --cov --cov-branch --cov-report=xml

   # Check code style
   poetry run black .
   poetry run flake8
   poetry run mypy tfbpmodeling
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Open a pull request against the `dev` branch
   - Provide clear description of changes
   - Reference any related issues

## Project Structure

```
tfbpmodeling/
├── tfbpmodeling/           # Main package
│   ├── __main__.py         # CLI entry point
│   ├── interface.py        # Main workflow functions
│   ├── modeling_input_data.py
│   ├── bootstrapped_input_data.py
│   ├── bootstrap_model_results.py
│   ├── evaluate_interactor_significance_*.py
│   ├── stratified_cv*.py
│   ├── utils/              # Utility functions
│   └── tests/              # Test suite
├── docs/                   # Documentation (MkDocs)
├── tmp/                    # Exploratory development
├── pyproject.toml          # Poetry configuration
├── mkdocs.yml             # Documentation configuration
├── CLAUDE.md              # Claude Code instructions
└── README.md
```

### Core Modules

- **`__main__.py`**: CLI entry point with argparse setup
- **`interface.py`**: Main workflow orchestration and CLI functions
- **`modeling_input_data.py`**: Data loading and preprocessing
- **`bootstrapped_input_data.py`**: Bootstrap resampling functionality
- **`bootstrap_model_results.py`**: Results aggregation and statistics
- **`evaluate_interactor_significance_*.py`**: Statistical significance testing
- **`stratified_cv*.py`**: Cross-validation with stratification
- **`utils/`**: Helper functions for data manipulation

### Exploratory Development

The `tmp/` directory is set up for exploratory data analysis and interactive development:

- **Jupyter notebooks**: Can be run from `tmp/` in the virtual environment
- **iPython kernel**: Installed in the development environment
- **Version control**: Files in `tmp/` are excluded from git tracking
- **Testing**: `tmp/` directory is ignored by pytest
- **Experimentation**: Safe space for prototyping and data exploration

See [tmp/README.md](../tmp/README.md) for more information about using this directory.

## Code Standards

### Style Guidelines

We use automated tools to maintain consistent code style:

- **Black**: Code formatting (88 character line length)
- **Flake8**: Style checking and linting
- **MyPy**: Type checking
- **isort**: Import sorting

### Code Quality

- **Type Hints**: All functions should have type hints
- **Docstrings**: Use Sphinx-style docstrings for all public functions
- **Tests**: Write tests for all new functionality
- **No Secrets**: Never commit API keys, passwords, or other secrets

### Example Code Style

```python
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def process_data(
    data: pd.DataFrame,
    threshold: float = 0.05,
    normalize: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    \"\"\"
    Process input data with filtering and normalization.

    :param data: Input dataframe with features as columns
    :param threshold: Minimum value threshold for filtering
    :param normalize: Whether to normalize data to unit variance
    :return: Processed dataframe and list of excluded features
    \"\"\"
    excluded_features: List[str] = []

    # Filter low-variance features
    for col in data.columns:
        if data[col].var() < threshold:
            excluded_features.append(col)

    processed_data = data.drop(columns=excluded_features)

    if normalize:
        processed_data = (processed_data - processed_data.mean()) / processed_data.std()

    return processed_data, excluded_features
```

## Testing

### Test Structure

Tests are located in `tfbpmodeling/tests/`:

```
tests/
├── test_interface.py
├── test_modeling_input_data.py
├── test_bootstrapped_input_data.py
├── test_utils.py
└── fixtures/
    ├── sample_expression.csv
    └── sample_binding.csv
```

### Writing Tests

Use pytest for all tests:

```python
import pytest
import pandas as pd
from tfbpmodeling.modeling_input_data import ModelingInputData


class TestModelingInputData:
    \"\"\"Test suite for ModelingInputData class.\"\"\"

    def test_initialization(self, sample_data_files):
        \"\"\"Test basic initialization.\"\"\"
        data = ModelingInputData(
            response_file=sample_data_files['response'],
            predictors_file=sample_data_files['predictors'],
            perturbed_tf='YPD1'
        )
        assert data is not None
        assert len(data.get_feature_names()) > 0

    def test_invalid_perturbed_tf(self, sample_data_files):
        \"\"\"Test error handling for invalid perturbed TF.\"\"\"
        with pytest.raises(KeyError, match="not found in response"):
            ModelingInputData(
                response_file=sample_data_files['response'],
                predictors_file=sample_data_files['predictors'],
                perturbed_tf='INVALID_TF'
            )

    @pytest.mark.parametrize("normalize", [True, False])
    def test_normalization_options(self, sample_data_files, normalize):
        \"\"\"Test normalization parameter.\"\"\"
        data = ModelingInputData(
            response_file=sample_data_files['response'],
            predictors_file=sample_data_files['predictors'],
            perturbed_tf='YPD1',
            normalize_weights=normalize
        )
        # Test that normalization was applied correctly
        assert data.normalize_weights == normalize


@pytest.fixture
def sample_data_files(tmp_path):
    \"\"\"Create sample data files for testing.\"\"\"
    # Create sample response data
    response_data = pd.DataFrame({
        'sample1': [0.1, 0.2, 0.3],
        'sample2': [0.4, 0.5, 0.6],
        'YPD1': [0.7, 0.8, 0.9]
    }, index=['gene1', 'gene2', 'gene3'])

    # Create sample predictor data
    predictor_data = pd.DataFrame({
        'TF1': [0.1, 0.2, 0.3],
        'TF2': [0.4, 0.5, 0.6]
    }, index=['gene1', 'gene2', 'gene3'])

    # Save to temporary files
    response_file = tmp_path / "response.csv"
    predictor_file = tmp_path / "predictors.csv"

    response_data.to_csv(response_file, index_label='gene_id')
    predictor_data.to_csv(predictor_file, index_label='gene_id')

    return {
        'response': str(response_file),
        'predictors': str(predictor_file)
    }
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tfbpmodeling/tests/test_interface.py

# Run with coverage
poetry run pytest --cov=tfbpmodeling --cov-report=html

# Run tests matching pattern
poetry run pytest -k "test_modeling"

# Run tests with verbose output
poetry run pytest -v
```

## Documentation

### Documentation Structure

Documentation is built with MkDocs and uses the Material theme:

```
docs/
├── index.md
├── getting-started/
│   ├── installation.md
│   └── quickstart.md
├── cli/
│   ├── overview.md
│   └── linear-perturbation-binding-modeling.md
├── tutorials/
│   ├── basic-workflow.md
│   └── advanced-features.md
├── api/
│   ├── interface.md
│   └── modeling_input_data.md
└── development/
    ├── contributing.md
    └── testing.md
```

### Writing Documentation

- Use clear, concise language
- Include code examples for all features
- Provide both basic and advanced usage examples
- Link between related sections

### Building Documentation

```bash
# Serve documentation locally with live reload
mkdocs serve

# Build documentation
mkdocs build

# Deploy to GitHub Pages (maintainers only)
poetry run mkdocs gh-deploy
```

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Environment Information**
   - Python version
   - tfbpmodeling version
   - Operating system

2. **Reproduction Steps**
   - Minimal code example
   - Input data characteristics
   - Expected vs actual behavior

3. **Error Messages**
   - Complete error traceback
   - Log output if available

### Feature Requests

For feature requests, provide:

1. **Use Case**: Describe the problem you're trying to solve
2. **Proposed Solution**: How you envision the feature working
3. **Alternatives**: Other approaches you've considered
4. **Examples**: Code examples of how it would be used

## Development Guidelines

### Branch Management

- **main**: Stable release branch
- **dev**: Development branch for integration
- **feature/***: Feature development branches
- **hotfix/***: Critical bug fixes

### Commit Messages

Use clear, descriptive commit messages:

```
Add bootstrap confidence interval calculation

- Implement percentile method for CI estimation
- Add support for custom confidence levels
- Include tests for edge cases
- Update documentation with examples
```

### Code Review Process

1. All changes require pull request review
2. CI tests must pass
3. Code coverage should not decrease
4. Documentation must be updated for new features
5. At least one maintainer approval required

### Release Process

1. Features merged to `dev` branch
2. Testing and integration on `dev`
3. Release candidate created
4. Final testing and documentation review
5. Merge to `main` and tag release
6. Update changelog and documentation

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Request Reviews**: Code-specific feedback

### Maintainer Contact

- **Chase Mateusiak**: Lead developer
- **Michael Brent**: Principal investigator

### Resources

- **[Project Documentation](https://brentlab.github.io/tfbpmodeling/)**
- **[GitHub Repository](https://github.com/BrentLab/tfbpmodeling)**
- **[Issue Tracker](https://github.com/BrentLab/tfbpmodeling/issues)**

## Troubleshooting

### Development Environment Issues

#### Poetry Installation Problems

If Poetry installation fails, try the alternative installation method:

```bash
pip install poetry
```

#### Virtual Environment Issues

If you encounter virtual environment problems:

```bash
# Remove existing environment
poetry env remove python

# Reinstall dependencies
poetry install
```

#### Pre-commit Hook Failures

If pre-commit hooks fail during commits:

```bash
# Run pre-commit manually to see specific issues
poetry run pre-commit run --all-files

# Fix any reported issues and commit again
```

#### Documentation Build Issues

If mkdocs build fails:

```bash
# Check for missing dependencies
poetry install

# Try building with verbose output
mkdocs build --verbose

# Check configuration
mkdocs config
```

#### Test Failures

If tests fail unexpectedly:

```bash
# Run tests with verbose output
poetry run pytest -v

# Run specific failing test
poetry run pytest path/to/failing_test.py::test_name -v

# Check test dependencies
poetry run pytest --collect-only
```

### Common Development Issues

#### Import Errors

```bash
# Ensure package is installed in development mode
poetry install

# Check Python path
python -c "import tfbpmodeling; print(tfbpmodeling.__file__)"
```

#### Module Not Found

```bash
# Verify virtual environment is activated
which python
poetry env info

# Reinstall in development mode
poetry install --no-deps
```

#### Permission Errors

```bash
# Check file permissions
ls -la

# Fix permissions if needed
chmod +x scripts/your_script.sh
```

### Getting Help with Development

If you encounter issues not covered here:

1. **Search existing issues**: Check if someone else has faced the same problem
2. **Create a detailed issue**: Include error messages, environment info, and steps to reproduce
3. **Join discussions**: Use GitHub Discussions for questions and help
4. **Contact maintainers**: Reach out directly for urgent issues

## Recognition

Contributors are recognized in:

- **CHANGELOG.md**: Feature and bug fix credits
- **AUTHORS.md**: Comprehensive contributor list
- **Release Notes**: Major contribution highlights
- **Documentation**: Author attributions for significant additions

Thank you for contributing to tfbpmodeling!