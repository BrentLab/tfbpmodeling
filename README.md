# tfbpmodeling

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![style](https://img.shields.io/badge/%20style-sphinx-0a507a.svg)](https://www.sphinx-doc.org/en/master/usage/index.html)
[![Pytest](https://github.com/BrentLab/tfbpmodeling/actions/workflows/ci.yml/badge.svg)](https://github.com/BrentLab/tfbpmodeling/actions/workflows/ci.yml)

## Documentation

See [here](https://brentlab.github.io/tfbpmodeling/) for more complete documentation

## Installation

This repo has not yet been added to PyPI. See the developer installation below.

### Development

1. If you plan on making pull requests, fork the repo to your own github account
1. git clone the repo (possibly your fork)
   * If you cloned your fork, add the BrentLab repo as your `upstream` remote

      ```bash
      git remote add upstream https://github.com/BrentLab/tfbpmodeling.git
      ```

1. `cd` into your local repo
1. `poetry install` to install the dependencies. If you don't have poetry, see the
   [poetry instructions](#poetry) below.
1. Install pre-commit hooks. If you do not have pre-commit,
  [install it](https://pre-commit.com/#install) first. Then run:

   ```bash
   pre-commit install
   ```

When adding/refactoring code, please make sure that you do so in a feature
branch that is branched from `dev`. Please keep your `dev` up to date by
rebasing it onto `main` after pulling from `upstream/main`. Similarly, keep
your feature branch up to date with dev by rebasing your `dev` branch onto
`upstream/dev`, and then rebasing your feature branch onto `dev`. This will
help avoid merge conflicts when you make a pull request.

When you are ready to make a pull request, please make sure that the pre-commit hooks
all pass locally. You can run pre-commit manually (before committing) with:

```bash
pre-commit run --all-files
```

if you properly installed the pre-commit hooks, pre-commit will automatically run
when you attempt to make a commit, also.

#### tmp/

There is a [tmp/] directory set up for exploratory data analysis and
interactive development. See the [tmp/README.md](tmp/README.md)
for more information. `ipykernel` is installed in the `dev` environment,
so jupyter notebooks can be run from `tmp/` in the virtual environment.

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
