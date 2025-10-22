#

# How to run a model with PCSE

To get familiar with WOFOST models and how to run the models, we recommend to
first check out the [PCSE
documentation](https://pcse.readthedocs.io/en/stable/index.html) and explore the
notebooks [01 Getting Started with
PCSE.ipynb](https://github.com/ajwdewit/pcse_notebooks/blob/master/01%20Getting%20Started%20with%20PCSE.ipynb)
and [02 Running with custom input data.ipynb
](https://github.com/ajwdewit/pcse_notebooks/blob/master/02%20Running%20with%20custom%20input%20data.ipynb).

In a nutshell, we can run a model, for example, `leaf_dynamics` using PCSE as:

```python
from diffwofost.physical_models.utils import EngineTestHelper

# create the model
model = EngineTestHelper(
    crop_parameters_provider,  # this provides the crop parameters
    weather_data_provider,
    agromanagement_provider,
    leaf_dynamics_config_file,  # this where the differentiable model is specified
    external_states,  # any external states if needed
)

# run the simulation with a fixed time step of one day
model.run_till_terminate()

# get the output
results = model.get_output()
```
