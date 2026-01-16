#

{% include-markdown "../README.md" end="# diffWOFOST" %}

## diffWOFOST: Differentiable WOFOST

The package diffWOFOST contains a differentiable implementation of the WOFOST
crop growth models using [`pytorch`](https://pytorch.org/) and
[`PCSE`](https://pcse.readthedocs.io/en/stable/index.html). The implementation
allows for automatic differentiation, enabling gradient-based optimization,
sensitivity analysis and data assimilation.

In PCSE, WOFOST models are categorized based on
`version, productionlevel, waterbalance, nitrogenbalance` and each model
contains a set of elements e.g. crop and soil models, see [models available in
PCSE](https://pcse.readthedocs.io/en/stable/available_models.html#models-available-in-pcse)
and [PCSE
Engine](https://pcse.readthedocs.io/en/stable/reference_guide.html#the-engine).

In diffWOFOST, each element is implemented as a differentiable module, using
[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html), allowing for
end-to-end differentiation of the entire WOFOST model. To develop a
differentiable module, we check for look-up tables, hard thresholds, and
mathematical operations, and replace them with differentiable alternatives.

In addition to differentiability, the implementation also focuses on efficiency,
by leveraging vectorized operations. This is particularly important for
large-scale simulations and training workflows, where the computational cost
can be significant.

## Hybrid modelling with diffWOFOST

Hybrid modelling, referring to a combination of process-based and
machine-learning modelling, has recently emerged as a promising line of research
to harness the strengths of both approaches while mitigating their respective
weaknesses, see [Integrating Scientific Knowledge with Machine Learning for
Engineering and Environmental Systems](
https://doi.org/10.48550/arXiv.2003.04919 ) and [Deep learning and process
understanding for data-driven Earth system
science](https://doi.org/10.1038/s41586-019-0912-1).

The approach where an machine learning (ML) model predicts physical parameters, which
are then used in a physics-based model, and combines both in a hybrid
architecture, is a state-of-the-art approach and is known under various names,
see [Scientific Machine
Learning](https://sciml.wur.nl/reviews/sciml/sciml.html). The mathematics would
be:

$$
\frac{\partial \text{loss}}{\partial \text{(ML model weights)}} = \frac{\partial \text{loss}}{\partial \text{(physics-based model output)}} \cdot \frac{\partial \text{(physics-based model output)}}{\partial \text{(physics-based model parameters)}} \cdot \frac{\partial \text{(physics-based model parameters)}}{\partial \text{(ML model weights)}}
$$

And code wise, this would look like:

```python
import torch.nn as nn

# Step 1: ML model that outputs physical parameters e.g. LSTM
class MLModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_physical_params):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_physical_params)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        physical_params = self.linear(lstm_out[:, -1, :])
        return physical_params

# Step 2: Physical model i.e. a differentiable WOFOST model e.g. Wofost72_PP
class PhysicalModel(nn.Module):
    def __init__(self, dt):
        super().__init__()

    def forward(self, params):
        model = Wofost72_PP(params, ...)  # this is differentiable version
        model.run_till_terminate()  # finish the simulation
        output = model.get_output()
        return output

# Step 3: Hybrid model integrating ML and physical model
class HybridModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_physical_params):
        super().__init__()
        self.ml_model = MLModel(input_size, hidden_size, num_physical_params)
        self.physical_model = PhysicalModel()

    def forward(self, x):
        physical_params = self.ml_model(x)
        output = self.physical_model(physical_params)
        return output, physical_params
```

## Code structure (under development)

The package is structured as follows:

```bash
├── physical_models/
        ├── crop/  # differentiable implementation of each crop model
        │   ├── leaf_dynamics.py
        │   ├── root_dynamics.py
        │   ├── ...
        ├── soil/
        ├── utils.py  # helpers
```

!!! note
    At the moment the package is under continuous development. So make sure that
    you install the latest version.
