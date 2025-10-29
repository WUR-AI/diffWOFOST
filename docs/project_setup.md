
Here we provide some details about the project setup. Most of the choices are
explained in the [Turing Way: Guide for Reproducible
Research](https://book.the-turing-way.org/reproducible-research/reproducible-research/).


## Repository structure

The repository has the following structure:

```bash

├── .github/        # GitHub specific files such as workflows
├── docs/           # Documentation source files
├── my_package/     # Main package code
├── tests/          # Test code
├── .gitignore      # Git ignore file
├── CITATION.cff    # Citation file
├── LICENSE         # License file
├── README.md       # User documentation
├── pyproject.toml  # Project configuration file and dependencies
├── mkdocs.yml      # MkDocs configuration file

```

## Package management and dependencies

You can use pip for installing dependencies and package
management.

- Runtime dependencies should be added to `pyproject.toml` in the `dependencies`
  list under `[project]`.
- Development dependencies, such as for testing or documentation, should be
  added to `pyproject.toml` in one of the lists under
  `[project.optional-dependencies]`.

## Packaging/One command install

You can distribute your code using PyPI. This can be done automatically using
GitHub workflows, see `.github/`.

## Package version number

- We recommend using [semantic versioning](https://semver.org/).
- For convenience, the package version is stored in a single place: `pyproject.toml`.
- Don't forget to update the version number before [making a release](./CONTRIBUTING.md)! Also, update `__version__` variable in `diffwofost/__init__.py` to the
   same version.

## CITATION.cff

- To allow others to cite your software, add a `CITATION.cff` file
- It only makes sense to do this once there is something to cite (e.g., a software release with a DOI).
- Follow the [Software Citation with CITATION.cff](https://book.the-turing-way.org/communication/citable/citable-cff/) section in the Turing Way guide.

## CODE_OF_CONDUCT.md

- Information about how to behave professionally
- To know more, read [Turing Way guide on Code of Conduct](https://book.the-turing-way.org/community-handbook/coc/)

## CONTRIBUTING.md

- Information about how to contribute to this software package
- To know more, read [Turing Way guide on Contributing](https://book.the-turing-way.org/reproducible-research/code-documentation/code-documentation-project/#contributing-guidelines)
