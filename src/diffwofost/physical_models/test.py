import pandas as pd
import torch
import yaml
from pcse import signals
from pcse.base.weather import WeatherDataContainer
from pcse.base.weather import WeatherDataProvider
from pcse.settings import settings
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.engine import Engine
from diffwofost.physical_models.parameter_providers import ParameterProvider
from diffwofost.physical_models.weather import to_weather_data_iterator


class EngineTestHelper(Engine):
    """An engine which is purely for running the YAML unit tests."""

    def _run(self):
        """Make one time step of the simulation."""
        # Update timer
        self.day, delt = self.timer()

        self.kiosk(self.day)
        # When the list of external states is exhausted, send crop_finish to
        # end the test run
        if self.kiosk.external_states_exhausted:
            self._send_signal(
                signal=signals.crop_finish, day=self.day, finish_type="maturity", crop_delete=False
            )

        # State integration and update to forced variables
        self.integrate(self.day, delt)

        # Driving variables
        self.drv = self._get_driving_variables(self.day)

        # Agromanagement decisions
        self.agromanager(self.day, self.drv)

        # Rate calculation
        self.calc_rates(self.day, self.drv)

        if self.flag_terminate is True:
            self._terminate_simulation(self.day)


class WeatherDataContainerTestHelper(WeatherDataContainer):
    """A helper class for creating WeatherDataContainer instances.

    It adds support for dict-like indexing of weather data, to provide compatibility with the
    interface used within diffWOFOST.
    """

    def __getitem__(self, key):
        return getattr(self, key)


class WeatherDataProviderTestHelper(WeatherDataProvider):
    """A helper class for creating a WeatherDataProvider instance.

    Needed to provide a data structure that can be used by both PCSE and diffWOFOST, useful for
    verifying the compatibility of diffWOFOST modules within PCSE.
    """

    def __init__(self, yaml_weather, meteo_range_checks=True):
        super().__init__()
        # This is a temporary workaround. The `METEO_RANGE_CHECKS` logic in
        # `__setattr__` method in `WeatherDataContainer` is not vector compatible
        # yet. So we can disable it here when creating the `WeatherDataContainer`
        # instances with arrays.
        settings.METEO_RANGE_CHECKS = meteo_range_checks
        for weather in yaml_weather:
            weather_inputs = {k: v for k, v in weather.items() if k != "SNOWDEPTH"}
            wdc = WeatherDataContainerTestHelper(**weather_inputs)
            self._store_WeatherDataContainer(wdc, wdc.DAY)


def prepare_engine_input(
    test_data,
    crop_model_params,
    device=None,
    dtype=None,
    return_weather_data_provider=False,  # set True for tests that patch PCSE
    meteo_range_checks=True,
):
    """Prepare the inputs for the engine from the YAML file."""
    # If not specified, use default dtype and device
    if device is None:
        device = ComputeConfig.get_device()
    if dtype is None:
        dtype = ComputeConfig.get_dtype()

    agro_management_inputs = test_data["AgroManagement"]
    cropd = test_data["ModelParameters"]

    weather_data = test_data["WeatherVariables"]
    if return_weather_data_provider:
        # If required, return the PCSE-compatible data structure for weather data
        weather_data_provider = WeatherDataProviderTestHelper(
            weather_data, meteo_range_checks=meteo_range_checks
        )
    else:
        weather_data_df = pd.DataFrame(weather_data)
        if "DTEMP" not in weather_data_df.columns:
            weather_data_df["DTEMP"] = (weather_data_df["TEMP"] + weather_data_df["TMAX"]) / 2.0
        weather_data_iterator = to_weather_data_iterator(weather_data_df, check=meteo_range_checks)

        # create a list out of the iterator, so that the weather data can be reused in several tests
        weather_data_provider = list(weather_data_iterator)

    crop_model_params_provider = ParameterProvider(cropdata=cropd)
    external_states = test_data.get("ExternalStates") or []

    # convert parameters to tensors
    crop_model_params_provider.clear_override()
    for name in crop_model_params:
        # if name is missing in the YAML, skip it
        if name in crop_model_params_provider:
            value = torch.tensor(crop_model_params_provider[name], dtype=dtype, device=device)
            crop_model_params_provider.set_override(name, value, check=False)

    # convert external states to tensors
    tensor_external_states = [
        {
            k: v if k == "DAY" else torch.tensor(v, dtype=dtype, device=device)
            for k, v in item.items()
        }
        for item in external_states
    ]
    return (
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        tensor_external_states,
    )


def get_test_data(test_data_path):
    """Get the test data from the YAML file."""
    with open(test_data_path) as f:
        return yaml.safe_load(f)


def calculate_numerical_grad(get_model_fn, param_name, param_value, out_name):
    """Calculate the numerical gradient of output with respect to a parameter."""
    delta = 1e-6

    # Parameters like RDRRTB are batched tables, so we need to compute
    # the gradient for each table element separately.
    # Flatten for easier indexing; clone once so we can restore in-place.
    param_flat = param_value.detach().reshape(-1).clone()
    grad_flat = torch.zeros_like(param_flat)

    with torch.no_grad():
        for i in range(param_flat.numel()):
            orig = param_flat[i].item()

            param_flat[i] = orig + delta
            model = get_model_fn()
            loss_plus = model({param_name: param_flat.view_as(param_value)})[out_name].sum()

            param_flat[i] = orig - delta
            model = get_model_fn()
            loss_minus = model({param_name: param_flat.view_as(param_value)})[out_name].sum()

            grad_flat[i] = (loss_plus - loss_minus) / (2 * delta)
            param_flat[i] = orig  # restore for next iteration

    return grad_flat.view_as(param_value)
