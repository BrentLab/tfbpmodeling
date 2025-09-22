# Installation

## Requirements

- Python 3.11 or higher
- Git for version control

## Standard Installation

tfbpmodeling is available for installation directly from GitHub using pip. PyPI distribution is planned for future releases.

### Install from GitHub

=== "Latest Stable (main branch)"

    ```bash
    python -m pip install git+https://github.com/BrentLab/tfbpmodeling.git
    ```

=== "Development Version (dev branch)"

    ```bash
    python -m pip install git+https://github.com/BrentLab/tfbpmodeling.git@dev
    ```

=== "Specific Version (by tag)"

    ```bash
    python -m pip install git+https://github.com/BrentLab/tfbpmodeling.git@v1.0.0
    ```

### Virtual Environment (Recommended)

It's recommended to install tfbpmodeling in a virtual environment:

```bash
# Create virtual environment
python -m venv tfbp-env

# Activate virtual environment
# On Linux/macOS:
source tfbp-env/bin/activate
# On Windows:
tfbp-env\Scripts\activate

# Install tfbpmodeling
python -m pip install git+https://github.com/BrentLab/tfbpmodeling.git

# Upgrade existing installation
python -m pip install --upgrade git+https://github.com/BrentLab/tfbpmodeling.git
```

## Verify Installation

Test that the installation was successful:

```bash
# Show help for the main command
python -m tfbpmodeling --help

# Show help for the modeling command
python -m tfbpmodeling linear_perturbation_binding_modeling --help
```

## Development Installation

For development work, you'll need to clone the repository and install with Poetry.

### 1. Clone the Repository

=== "Fork for Development"

    If you plan to contribute:

    1. Fork the repository on GitHub
    2. Clone your fork:

    ```bash
    git clone https://github.com/YOUR_USERNAME/tfbpmodeling.git
    cd tfbpmodeling
    ```

    3. Add the upstream remote:

    ```bash
    git remote add upstream https://github.com/BrentLab/tfbpmodeling.git
    ```

=== "Direct Clone"

    ```bash
    git clone https://github.com/BrentLab/tfbpmodeling.git
    cd tfbpmodeling
    ```

### 2. Install Poetry

If you don't have Poetry installed, follow the [official installation instructions](https://python-poetry.org/docs/#installation).

!!! tip "Poetry Configuration"
    We recommend configuring Poetry to create virtual environments in the project directory:

    ```bash
    poetry config virtualenvs.in-project true
    ```

    This creates the virtual environment as `.venv` in the project directory.

### 3. Install Dependencies

```bash
poetry install
```

This installs all dependencies, including development tools for testing and documentation.

### 4. Install Pre-commit Hooks

For development work, install pre-commit hooks to ensure code quality:

```bash
poetry run pre-commit install
```

You can run pre-commit manually at any time:

```bash
poetry run pre-commit run --all-files
```

### Development Tools

#### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov --cov-branch --cov-report=xml

# Run specific test file
poetry run pytest tfbpmodeling/tests/test_interface.py

# Run specific test by name
poetry run pytest -k "test_function_name"
```

#### Code Quality

The project uses several tools to maintain code quality:

```bash
# Format code with Black
poetry run black .

# Check code style with Flake8
poetry run flake8

# Type checking with MyPy
poetry run mypy tfbpmodeling
```

#### Documentation

Build and serve documentation locally:

```bash
# Serve documentation with live reload
mkdocs serve

# Build documentation
mkdocs build

# Deploy to GitHub Pages (maintainers only)
poetry run mkdocs gh-deploy
```

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/BrentLab/tfbpmodeling/issues)
- **GitHub Discussions**: [Ask questions or discuss usage](https://github.com/BrentLab/tfbpmodeling/discussions)
- **Documentation**: Browse the complete documentation for detailed usage examples
- **Development Guide**: See [Contributing](../development/contributing.md) for development setup and troubleshooting