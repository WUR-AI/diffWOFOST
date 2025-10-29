#

If you are a developer wanting to contribute to the diffwofost package, this
guide will help you get started. First check out the contribution guidelines in
[Contributing guide](CONTRIBUTING.md) and the [Project setup](project_setup.md)
to get familiar with the package structure and development practices.

## Installation in development mode

To install the package in development mode, you need to clone the source code
and install the package in development mode:

```bash
git clone https://github.com/WUR-AI/diffWOFOST.git
cd diffWOFOST
pip install -e .[dev]  ## install with development dependencies
pip install -e .[docs]  ## install with documentation dependencies
```

## GitHub collaboration workflow

We use a GitHub collaboration workflow based on feature branches and pull
requests. When starting adding a new feature or fixing a bug, create a new
branch from `main` branch. When your changes are ready, create a pull request to
merge your changes back into `main` branch. Make sure to ask for at least one
review from another team member before merging your pull request.

## Running the tests

- Tests should be put in the `tests` folder.
- The testing framework used is [PyTest](https://pytest.org)
- The project uses [GitHub action workflows](https://docs.github.com/en/actions)
  to automatically run tests on GitHub infrastructure against multiple Python
  versions. Workflows can be found in `.github/workflows` directory.
- [Relevant section in the
  guide](https://guide.esciencecenter.nl/#/best_practices/language_guides/python?id=testing)
- To run the tests locally, you need to make sure that you have installed the
development dependencies as described in the [Installation in development
mode](#installation-in-development-mode) section.
Then, inside the package directory, run:

```bash
pytest -v
```

to run all tests with verbose output. To run an individual test file, run:

```bash
pytest -v tests/test_my_module.py
```

### Linters

For linting and sorting imports we will use [ruff](https://beta.ruff.rs/docs/).
Running the linters requires an activated virtual environment with the
development tools installed.

```shell
# linter
ruff check .

# linter with automatic fixing
ruff check . --fix

# check formatting only
ruff format --check . --diff
```

## Documentation page

- Documentation should be put in the `docs/` directory.
- We recommend writing the documentation using Google style docstrings.
- The documentation is set up with the [MkDocs](https://www.mkdocs.org/).
  - `.mkdocs.yml` is the [MkDocs](https://www.mkdocs.org/) configuration file. When MkDocs is building the documentation this package and its development dependencies are installed so the API reference can be rendered.
- Make sure you have installed the documentation dependencies as described in the
[Installation in development mode](#installation-in-development-mode) section.
Then, inside the package directory, run:

```bash

# Build the documentation
mkdocs build

# Preview the documentation
mkdocs serve

```

Click on the link provided in the terminal to view the documentation page.

## Coding style conventions and code quality

- [Relevant section in the NLeSC guide](https://guide.esciencecenter.nl/#/best_practices/language_guides/python?id=coding-style-conventions).

## Continuous code quality

[Sonarcloud](https://sonarcloud.io/) is used to perform quality analysis and code coverage report

- `sonar-project.properties` is the SonarCloud [configuration](https://docs.sonarqube.org/latest/analysis/analysis-parameters/) file
- `.github/workflows/sonarcloud.yml` is the GitHub action workflow which performs the SonarCloud analysis