# Differentiable WOFOST

The package diffwofost contains a differentiable implementation of the WOFOST
crop growth models using [`pytorch`](https://pytorch.org/) and
[`PCSE`](https://pcse.readthedocs.io/en/stable/index.html). The implementation
allows for automatic differentiation, enabling gradient-based optimization and
sensitivity analysis.

In PCSE, WOFOST models are categorized based on
`<version>_<productionlevel>_<waterbalance>_<nitrogenbalance>` and ech model
contains a set of submodels, see [Models available in
PCSE](https://pcse.readthedocs.io/en/stable/available_models.html#models-available-in-pcse)
and [PCSE
Engine](https://pcse.readthedocs.io/en/stable/reference_guide.html#the-engine).
in diffwofost, each submodel is implemented as a differentiable pytorch module,
allowing for end-to-end differentiation of the entire WOFOST model, see
[torch.nn](https://docs.pytorch.org/docs/stable/nn.html).
