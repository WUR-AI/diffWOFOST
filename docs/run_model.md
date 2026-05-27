#

# How to run a model with PCSE

To get familiar with WOFOST models and how to run the models, we recommend to
first check out the [PCSE
documentation](https://pcse.readthedocs.io/en/stable/index.html) and explore the
notebooks [01 Getting Started with
PCSE.ipynb](https://github.com/ajwdewit/pcse_notebooks/blob/master/01%20Getting%20Started%20with%20PCSE.ipynb)
and [02 Running with custom input data.ipynb
](https://github.com/ajwdewit/pcse_notebooks/blob/master/02%20Running%20with%20custom%20input%20data.ipynb).

In a nutshell, we can run a model, for example, `leaf_dynamics` using diffWOFOST as:

```python
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.leaf_dynamics import WOFOST_Leaf_Dynamics

# create config
leaf_dynamics_config = Configuration(
    CROP=WOFOST_Leaf_Dynamics,
    OUTPUT_VARS=["LAI", "TWLV"],
)

# create the model
model = EngineTestHelper(
    crop_parameters_provider,  # this provides the crop parameters
    weather_data_provider,
    agromanagement_provider,
    leaf_dynamics_config,  # this where the differentiable model is specified
    external_states,  # any external states if needed
)

# run the simulation with a fixed time step of one day
model.run_till_terminate()

# get the output
results = model.get_output()
```

See the notebooks in the [examples](./examples.md) section for more details and
examples on how to run the models.

## How to set computing device and data type

By default, the model will run on torch default device and dtype i.e.
`torch.get_default_device()` and `torch.get_default_dtype()`. See the
[examples](./examples.md) section for more details.

## How to run a crop model (physical) with torch.nn.Module

The crop models (physical) in diffWOFOST are implemented as a
`SimulationObject`, and they can be run using `Engine`, as explained above. To
run a model using `torch.nn.Module`, you can simply create a wrapper class that
inherits from `torch.nn.Module` and calls the `Engine.setup()` in the `forward`
method. See the notebooks in the [examples](./examples.md) section for more
details and examples on how to run the models with `torch.nn.Module`.

## How to run a crop model (ml-based)

The crop models (ml-based) in diffWOFOST are also implemented as a
`SimulationObject`, and they can be run using `Engine`, as explained above. To
run a model, you can simply specify the class and any ml-based model that class
uses in the `Configuration` when creating the `Engine` as:

```python
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.config import Configuration
from diffwofost.ml_models.crop.partitioning import DVS_Partitioning_NN, PartitioningMLP


# create config
partition_config = Configuration(
    CROP=DVS_Partitioning_NN,
    CROP_NN_MODEL=PartitioningMLP(hidden_size=32),  # as an example
    OUTPUT_VARS=["LAI", "TWLV"],
)

# create the model
model = EngineTestHelper(
    crop_parameters_provider,  # this provides the crop parameters
    weather_data_provider,
    agromanagement_provider,
    partition_config,  # this where the ml-based model is specified
    external_states,  # any external states if needed
)

# run the simulation with a fixed time step of one day
model.run_till_terminate()

# get the output
results = model.get_output()
```

## How to replace a model with a machine learning model (Hybrid modeling)

The crop (ml-based) models in diffWOFOST are also implemented as a
`SimulationObject`. The physical computations inside the class can be replaced
with a machine learning model. See the source codes in `ml_models/crop/` for
examples of how to replace a physical model with a machine learning model, such
as the partitioning model in WOFOST72. Also, there are notebooks in the
[examples](./examples.md) section for more details and examples on how to
replace the `partitioning` model with a machine learning model in `wofost72`.
