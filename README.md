<!-- (Customize these badges with your own links, and check https://shields.io/ or https://badgen.net/ to see which other badges are available.) -->

[![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/WUR-AI/diffWOFOST)
[![PyPI - Version](https://badge.fury.io/py/diffwofost.svg)](https://img.shields.io/pypi/v/diffwofost)
[![Python package built](https://github.com/WUR-AI/diffWOFOST/actions/workflows/build.yml/badge.svg)](https://github.com/WUR-AI/diffWOFOST/actions/workflows/build.yml)
[![Documentation built](https://github.com/WUR-AI/diffWOFOST/actions/workflows/deploy-documentation.yml/badge.svg)](https://github.com/WUR-AI/diffWOFOST/actions/workflows/deploy-documentation.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=WUR-AI_diffWOFOST&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=WUR-AI_diffWOFOST)

# diffWOFOST

<img align="right" width="90" alt="Logo" src="./docs/logo/diffwofost.png">

The python package `diffWOFOST` is a differentiable implementation of WOFOST models using [`torch`](https://pytorch.org/),
allowing gradients to flow through the simulations for optimization and data assimilation.

## Installation

You can install `diffWOFOST` using pip:

```bash
pip install diffwofost
```

To install the package in development mode, you can clone the repository and
install it using pip:

```bash
pip install -e .[dev]
```

To work with notebooks, you need to install `jupyterlab`:

```bash
pip install jupyterlab
```

## Documentation

The documentation for `diffWOFOST` is available at
[https://WUR-AI.github.io/diffWOFOST](https://WUR-AI.github.io/diffWOFOST).

## Acknowledgements

The package `diffWOFOST` is developed in the project
[deltaCrop](https://research-software-directory.org/projects/deltacrop), a
collaboration between Wageningen University & Research and Netherlands eScience
Center.
