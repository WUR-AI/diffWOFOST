# diffWOFOST

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

## Usage

An example notebooks are provided in the `docs/notebooks` folder.

## PCSE

The python implementation of WOFOST is available at
[`PCSE`](https://pcse.readthedocs.io/en/stable/). For more information about the
models, the functional components of PCSE, and documentation have a look at the
following links:

- [Models available in PCSE](https://pcse.readthedocs.io/en/stable/available_models.html#models-available-in-pcse)
- [The Engine](https://pcse.readthedocs.io/en/stable/reference_guide.html#the-engine)
- [PCSE source code](https://github.com/ajwdewit/pcse)
- [PCSE test data](https://github.com/ajwdewit/pcse/tree/master/tests/test_data)
- [PCSE example notebooks](https://github.com/ajwdewit/pcse_notebooks).
- [WOFOST_crop_parameters](https://github.com/ajwdewit/WOFOST_crop_parameters): each branch contains the crop parameters for a specific crop model version
