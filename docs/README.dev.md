# `diffwofost` developer documentation

If you're looking for user documentation, go [here](README.md).

## Development install

```shell
# Create a virtual environment, e.g. with
python -m venv env

# activate virtual environment
source env/bin/activate

# make sure to have a recent version of pip and setuptools
python -m pip install --upgrade pip setuptools

# (from the project root directory)
# install diffwofost as an editable package
python -m pip install --no-cache-dir --editable .
# install development dependencies
python -m pip install --no-cache-dir --editable .[dev]
# install documentation dependencies only
python -m pip install --no-cache-dir --editable .[docs]
```

Afterwards check that the install directory is present in the `PATH` environment variable.

## Running the tests

There are two ways to run tests.

The first way requires an activated virtual environment with the development tools installed:

```shell
pytest -v
```

### Test coverage

In addition to just running the tests to see if they pass, they can be used for coverage statistics, i.e. to determine how much of the package's code is actually executed during tests.
In an activated virtual environment with the development tools installed, inside the package directory, run:

```shell
coverage run
```

This runs tests and stores the result in a `.coverage` file.
To see the results on the command line, run

```shell
coverage report
```

`coverage` can also generate output in HTML and other formats; see `coverage help` for more information.## Running linters locally

For linting and sorting imports we will use [ruff](https://beta.ruff.rs/docs/). Running the linters requires an
activated virtual environment with the development tools installed.

```shell
# linter
ruff check .

# linter with automatic fixing
ruff check . --fix
```

To fix readability of your code style you can use [yapf](https://github.com/google/yapf)

## Generating documentation page

- Install the required dependencies as:

  ```bash
  cd FOWT-ML
  pip install -e .[docs]
  ```

- Build the documentation as:

  ```bash
  mkdocs build
  ```

- Preview the documentation as:

  ```bash
  mkdocs serve
  ```

Click on the link provided in the terminal to view the documentation page.

## Versioning

Bumping the version across all files is done with [bump-my-version](https://github.com/callowayproject/bump-my-version), e.g.

```shell
bump-my-version bump major  # bumps from e.g. 0.3.2 to 1.0.0
bump-my-version bump minor  # bumps from e.g. 0.3.2 to 0.4.0
bump-my-version bump patch  # bumps from e.g. 0.3.2 to 0.3.3
```

## Making a release

This section describes how to make a release in 3 parts:

1. preparation
1. making a release on PyPI
1. making a release on GitHub

### (1/3) Preparation


1. Verify that the information in [`CITATION.cff`](CITATION.cff) is correct.
1. Make sure the [version has been updated](#versioning).
1. Run the unit tests with `pytest -v`

### (2/3) PyPI

In a new terminal:

```shell
# OPTIONAL: prepare a new directory with fresh git clone to ensure the release
# has the state of origin/main branch
cd $(mktemp -d diffwofost.XXXXXX)
git clone git@github.com:WUR-AI/diffwofost .

# make sure to have a recent version of pip and the publishing dependencies
python -m pip install --upgrade pip
python -m pip install .[publishing]

# create the source distribution and the wheel
python -m build

# upload to test pypi instance (requires credentials)
python -m twine upload --repository testpypi dist/*
```

Visit
[https://test.pypi.org/project/diffwofost](https://test.pypi.org/project/diffwofost)
and verify that your package was uploaded successfully. Keep the terminal open, we'll need it later.

In a new terminal, without an activated virtual environment or an env directory:

```shell
cd $(mktemp -d diffwofost-test.XXXXXX)

# prepare a clean virtual environment and activate it
python -m venv env
source env/bin/activate

# make sure to have a recent version of pip and setuptools
python -m pip install --upgrade pip

# install from test pypi instance:
python -m pip -v install --no-cache-dir \
--index-url https://test.pypi.org/simple/ \
--extra-index-url https://pypi.org/simple diffwofost
```

Check that the package works as it should when installed from pypitest.

Then upload to pypi.org with:

```shell
# Back to the first terminal,
# FINAL STEP: upload to PyPI (requires credentials)
python -m twine upload dist/*
```

### (3/3) GitHub

Don't forget to also make a [release on GitHub](https://github.com/WUR-AI/diffwofost/releases/new).GitHub-Zenodo integration will also trigger Zenodo into making a snapshot of your repository and sticking a DOI on it.