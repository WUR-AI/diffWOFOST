#

## Optimization with diffWOFOST

We provide an example notebook showing optimization of models' parameters with
`diffWOFOST`. To get familiar with the concepts and implementation, check out
[`Introduction`](./index.md) in the documentation.

| Model | Open the notebook | Access the source | View the notebook |
|---|----|------------|---------------|
| Phenology | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][pheno_colab_link] | [![Access the source code](https://img.shields.io/badge/GitHub_Repository-000.svg?logo=github&labelColor=gray&color=blue)][pheno_source_link] | [![here](https://img.shields.io/badge/View_Notebook-orange.svg?logo=jupyter&labelColor=gray)](./notebooks/optimization_phenology.ipynb) |
| Root dynamics| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][root_colab_link] | [![Access the source code](https://img.shields.io/badge/GitHub_Repository-000.svg?logo=github&labelColor=gray&color=blue)][root_source_link] | [![here](https://img.shields.io/badge/View_Notebook-orange.svg?logo=jupyter&labelColor=gray)](./notebooks/optimization_root_dynamics.ipynb) |
| Leaf dynamics| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][leaf_colab_link] | [![Access the source code](https://img.shields.io/badge/GitHub_Repository-000.svg?logo=github&labelColor=gray&color=blue)][leaf_source_link] | [![here](https://img.shields.io/badge/View_Notebook-orange.svg?logo=jupyter&labelColor=gray)](./notebooks/optimization_leaf_dynamics.ipynb) |


!!! note

    When calculating gradients, it is important to ensure that the predicted
    physical parameters are within realistic bounds regarding the crop and
    environmental conditions.

    Also, when calculating gradients of an output w.r.t. parameters, it would be
    good to know in advance how the parameters in a model influence the outputs.
    If a parameter has little to no influence on an output, the gradient of the
    output w.r.t the parameter will be close to zero, which may not provide
    useful information for optimization.


## Computing configuration

The object `ComputeConfig` provides a central configuration for PyTorch device
and dtype settings across all simulation objects in diffWOFOST. Instead of
setting device and dtype individually for each class, use this central
configuration to apply settings globally.

**Default Behavior:**

- **Device**: Automatically defaults to 'cuda' if available, otherwise 'cpu'
- **Dtype**: Defaults to torch.float64

**Basic Usage:**

```python
from diffwofost.physical_models.config import ComputeConfig
import torch
# Set device to CPU
ComputeConfig.set_device('cpu')

# Or use a torch.device object
ComputeConfig.set_device(torch.device('cuda'))

# Set dtype to float32
ComputeConfig.set_dtype(torch.float32)

# Get current settings
device = ComputeConfig.get_device()  # Returns: torch.device('cpu')
dtype = ComputeConfig.get_dtype()    # Returns: torch.float32
```

**More info:**

See the [ComputeConfig API
reference](./api_reference.md#diffwofost.physical_models.config.ComputeConfig)
for more details.

[leaf_colab_link]: https://colab.research.google.com/github/WUR-AI/diffWOFOST/blob/main/docs/notebooks/optimization_leaf_dynamics.ipynb
[leaf_source_link]: https://github.com/WUR-AI/diffWOFOST/blob/main/docs/notebooks/optimization_leaf_dynamics.ipynb
[root_colab_link]: https://colab.research.google.com/github/WUR-AI/diffWOFOST/blob/main/docs/notebooks/optimization_root_dynamics.ipynb
[root_source_link]: https://github.com/WUR-AI/diffWOFOST/blob/main/docs/notebooks/optimization_root_dynamics.ipynb
[pheno_colab_link]: https://colab.research.google.com/github/WUR-AI/diffWOFOST/blob/main/docs/notebooks/optimization_phenology.ipynb
[pheno_source_link]: https://github.com/WUR-AI/diffWOFOST/blob/main/docs/notebooks/optimization_phenology.ipynb
