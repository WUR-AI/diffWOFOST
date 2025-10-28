#

## Optimization with diffWOFOST

We provide an example notebook showing optimization of models' parameters with
`diffWOFOST`. To get familiar with the concepts and implementation, check out
[`Introduction`](./index.md) in the documentation.

| Open the notebook | Access the source | View the notebook |
|-------|------------|---------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)][colab_link] | [![Access the source code](https://img.shields.io/badge/GitHub_Repository-000.svg?logo=github&labelColor=gray&color=blue)][source_link] | [![here](https://img.shields.io/badge/View_Notebook-orange.svg?logo=jupyter&labelColor=gray)](./notebooks/optimization.ipynb) |

!!! note
    When calculating gradients, it is important to ensure that the predicted
    physical parameters are within realistic bounds regarding the crop and
    environmental conditions.

    Also, when calculating gradients of an output w.r.t. parameters, it would be
    good to know in advance how the parameters in a model influence the outputs.
    If a parameter has little to no influence on an output, the gradient of the
    output w.r.t the parameter will be close to zero, which may not provide
    useful information for optimization.

[colab_link]: https://colab.research.google.com/github/WUR-AI/diffWOFOST/blob/main/docs/notebooks/optimization.ipynb

[source_link]: https://github.com/WUR-AI/diffWOFOST/blob/main/docs/notebooks/optimization.ipynb
