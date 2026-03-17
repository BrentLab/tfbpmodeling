# Contributing to tfbpmodeling

We welcome contributions to tfbpmodeling! This guide will help you get started with contributing code, documentation, or bug reports.

## Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/tfbpmodeling.git
   cd tfbpmodeling
   git remote add upstream https://github.com/BrentLab/tfbpmodeling.git
   ```

2. **Install Dependencies**
   ```bash
   pip install poetry
   poetry config virtualenvs.in-project true
   poetry install
   ```

3. **Set Up Pre-commit Hooks**
   ```bash
   poetry run pre-commit install
   ```

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout dev
   git pull upstream dev
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**, then commit — pre-commit hooks handle formatting and linting automatically:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request** against the `dev` branch with a clear description of changes.

## Project Structure

```
tfbpmodeling/
├── tfbpmodeling/           # Main package
│   ├── __main__.py         # CLI entry point and main workflow
│   ├── modeling_input_data.py
│   ├── bootstrapped_input_data.py
│   ├── bootstrap_model_results.py
│   ├── evaluate_interactor_significance_*.py
│   ├── stratified_cv*.py
│   ├── configure_logger.py
│   ├── utils/              # Utility functions
│   └── tests/              # Test suite
├── docs/                   # Documentation (MkDocs)
├── tmp/                    # Exploratory development (git-ignored)
├── pyproject.toml
├── mkdocs.yml
├── CLAUDE.md
└── README.md
```

The `tmp/` directory is available for exploratory work — notebooks, scratch scripts, etc. It is excluded from git and ignored by pytest.

## Code Standards

- **Black**: code formatting (88 character line length)
- **Flake8**: style checking
- **MyPy**: type checking
- **Type hints** on all public functions
- **Sphinx-style docstrings** on all public functions
- Never commit secrets or credentials

## Testing

```bash
# Run all tests
poetry run pytest

# Run a specific file or test
poetry run pytest tfbpmodeling/tests/test_interface.py
poetry run pytest -k "test_linear_workflow_logs"

# Run with coverage
poetry run pytest --cov --cov-branch --cov-report=xml
```

Every module should be reachable from the test suite. The goal is not exhaustive correctness proofs — a test that simply exercises a code path is sufficient. This ensures that any developer can set a breakpoint anywhere in the codebase, run `pytest` in debug mode, and be guaranteed to hit it.

## Documentation

```bash
mkdocs serve       # live preview
mkdocs build       # build static site
poetry run mkdocs gh-deploy  # deploy to GitHub Pages (maintainers only)
```

## Issue Reporting

When reporting bugs, include: Python version, tfbpmodeling version, OS, minimal reproduction steps, and the full error traceback.

For feature requests, describe the problem you're solving, your proposed solution, and a usage example.

## Branch Management

- **main**: stable releases
- **dev**: integration branch — PRs target here
- **feature/***: feature development
- **hotfix/***: critical fixes

## Getting Help

- **GitHub Issues**: bugs and feature requests
- **GitHub Discussions**: questions and general discussion
- **Chase Mateusiak**: lead developer
- **Michael Brent**: principal investigator
