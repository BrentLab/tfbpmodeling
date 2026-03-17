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
python -m tfbpmodeling --help
```

## Development Installation

See [Contributing](../development/contributing.md) for development setup instructions.

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/BrentLab/tfbpmodeling/issues)
- **GitHub Discussions**: [Ask questions or discuss usage](https://github.com/BrentLab/tfbpmodeling/discussions)