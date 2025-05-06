# tfbpmodeling

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![style](https://img.shields.io/badge/%20style-sphinx-0a507a.svg)](https://www.sphinx-doc.org/en/master/usage/index.html)
[![Pytest](https://github.com/BrentLab/tfbpmodeling/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BrentLab/tfbpmodeling/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BrentLab/tfbpmodeling/graph/badge.svg?token=D2AB7IUY7F)](https://codecov.io/gh/BrentLab/tfbpmodeling)

## Documentation

See [here](https://brentlab.github.io/tfbpmodeling/) for more complete documentation

## Installation

This repo has not yet been added to PyPI. See the developer installation below.

### Development

1. If you plan on making pull requests, fork the repo to your own github account
1. git clone the repo (possibly your fork)
   * Add the BrentLab repo as your `upstream` remote

      ```bash
      git remote add upstream https://github.com/BrentLab/tfbpmodeling.git
      ```

1. `cd` into the local version of the repo
1. `poetry install` to install the dependencies. If you don't have poetry, see the
   [poetry instructions][#poetry] below.

#### poetry

To install poetry, follow the instructions [here](https://python-poetry.org/docs/#installation).

I prefer setting the following configuration option so that
virtual environments are installed in the project directory:

```bash
poetry config virtualenvs.in-project true
```

So that the virtual environments are installed in the project directory as `.venv`

After cloning and `cd`ing into the repo, you can install the dependencies with:

```bash
poetry install
```

#### mkdocs

The documentation is build with mkdocs:

##### Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

##### Project layout

* mkdocs.yml    # The configuration file.
* docs/
   * index.md  # The documentation homepage.
   * ...       # Other markdown pages, images and other files.

To update the gh-pages documentation, use `poetry run mkdocs gh-deply`
