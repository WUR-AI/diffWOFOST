"""This file contains code that is required to run the YAML unit tests.

It contains:
    - VariableKioskTestHelper: A subclass of the VariableKiosk that can use externally
      forced states/rates
    - ConfigurationLoaderTestHelper: An subclass of ConfigurationLoader that allows to
      specify the simbojects to be test dynamically
    - EngineTestHelper: engine specifically for running the YAML tests.
    - WeatherDataProviderTestHelper: a weatherdata provides that takes the weather
      inputs from the YAML file.

Note that the code here is *not* python2 compatible.
"""

import logging
import os
from collections.abc import Iterable
import torch
import yaml
from pcse import signals
from pcse.agromanager import AgroManager
from pcse.base import ConfigurationLoader
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.base.weather import WeatherDataContainer
from pcse.base.weather import WeatherDataProvider
from pcse.engine import BaseEngine
from pcse.engine import Engine
from pcse.settings import settings
from pcse.timer import Timer
from pcse.traitlets import TraitType

DTYPE = torch.float64  # Default data type for tensors in this module

logging.disable(logging.CRITICAL)

this_dir = os.path.dirname(__file__)


def nothing(*args, **kwargs):
    """A function that does nothing."""
    pass


class VariableKioskTestHelper(VariableKiosk):
    """Variable Kiosk for testing purposes which allows to use external states."""

    external_state_list = None

    def __init__(self, external_state_list):
        super().__init__()
        self.current_externals = {}
        if external_state_list is not None:
            self.external_state_list = external_state_list

    def __call__(self, day):
        """Sets the external state/rate variables for the current day.

        Returns True if the list of external state/rate variables is exhausted,
        otherwise False.
        """
        if self.external_state_list is not None:
            current_externals = self.external_state_list.pop(0)
            forcing_day = current_externals.pop("DAY")
            msg = "Failure updating VariableKiosk with external states: days are not matching!"
            assert forcing_day == day, msg
            self.current_externals.clear()
            self.current_externals.update(current_externals)
            if len(self.external_state_list) == 0:
                return True

        return False

    def is_external_state(self, item):
        """Returns True if the item is an external state."""
        return item in self.current_externals

    def __getattr__(self, item):
        """Allow use of attribute notation.

        eg "kiosk.LAI" on published rates or states.
        """
        if item in self.current_externals:
            return self.current_externals[item]
        else:
            return dict.__getitem__(self, item)

    def __getitem__(self, item):
        """Override __getitem__ to first look in external states."""
        if item in self.current_externals:
            return self.current_externals[item]
        else:
            return dict.__getitem__(self, item)

    def __contains__(self, key):
        """Override __contains__ to first look in external states."""
        return key in self.current_externals or dict.__contains__(self, key)


class ConfigurationLoaderTestHelper(ConfigurationLoader):
    def __init__(self, YAML_test_inputs, simobject, waterbalance=None):
        self.model_config_file = "Test config"
        self.description = "Configuration loader for running YAML tests"
        self.CROP = simobject
        self.SOIL = waterbalance
        self.AGROMANAGEMENT = AgroManager
        self.OUTPUT_INTERVAL = "daily"
        self.OUTPUT_INTERVAL_DAYS = 1
        self.OUTPUT_WEEKDAY = 0
        self.OUTPUT_VARS = list(YAML_test_inputs["Precision"].keys())
        self.SUMMARY_OUTPUT_VARS = []
        self.TERMINAL_OUTPUT_VARS = []


class EngineTestHelper(Engine):
    """An engine which is purely for running the YAML unit tests."""

    def __init__(
        self,
        parameterprovider,
        weatherdataprovider,
        agromanagement,
        test_config,
        external_states=None,
    ):
        BaseEngine.__init__(self)

        # Load the model configuration
        self.mconf = ConfigurationLoader(test_config)
        self.parameterprovider = parameterprovider

        # Variable kiosk for registering and publishing variables
        self.kiosk = VariableKioskTestHelper(external_states)

        # Placeholder for variables to be saved during a model run
        self._saved_output = list()
        self._saved_summary_output = list()
        self._saved_terminal_output = dict()

        # register handlers for starting/finishing the crop simulation, for
        # handling output and terminating the system
        self._connect_signal(self._on_CROP_START, signal=signals.crop_start)
        self._connect_signal(self._on_CROP_FINISH, signal=signals.crop_finish)
        self._connect_signal(self._on_OUTPUT, signal=signals.output)
        self._connect_signal(self._on_TERMINATE, signal=signals.terminate)

        # Component for agromanagement
        self.agromanager = self.mconf.AGROMANAGEMENT(self.kiosk, agromanagement)
        start_date = self.agromanager.start_date
        end_date = self.agromanager.end_date

        # Timer: starting day, final day and model output
        self.timer = Timer(self.kiosk, start_date, end_date, self.mconf)
        self.day, delt = self.timer()
        # Update external states in the kiosk
        self.kiosk(self.day)

        # Driving variables
        self.weatherdataprovider = weatherdataprovider
        self.drv = self._get_driving_variables(self.day)

        # Component for simulation of soil processes
        if self.mconf.SOIL is not None:
            self.soil = self.mconf.SOIL(self.day, self.kiosk, parameterprovider)

        # Call AgroManagement module for management actions at initialization
        self.agromanager(self.day, self.drv)

        # Calculate initial rates
        self.calc_rates(self.day, self.drv)

    def _run(self):
        """Make one time step of the simulation."""
        # Update timer
        self.day, delt = self.timer()

        # When the list of external states is exhausted the VariableKioskTestHelper will
        # return True signalling the end of the test
        stop_test = self.kiosk(self.day)
        if stop_test:
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


class WeatherDataProviderTestHelper(WeatherDataProvider):
    """It stores the weatherdata contained within the YAML tests."""

    def __init__(self, yaml_weather, meteo_range_checks=True):
        super().__init__()
        # This is a temporary workaround. The `METEO_RANGE_CHECKS` logic in
        # `__setattr__` method in `WeatherDataContainer` is not vector compatible
        # yet. So we can disable it here when creating the `WeatherDataContainer`
        # instances with arrays.
        settings.METEO_RANGE_CHECKS = meteo_range_checks
        for weather in yaml_weather:
            if "SNOWDEPTH" in weather:
                weather.pop("SNOWDEPTH")
            wdc = WeatherDataContainer(**weather)
            self._store_WeatherDataContainer(wdc, wdc.DAY)


def prepare_engine_input(
    test_data, crop_model_params, meteo_range_checks=True, dtype=torch.float64
):
    """Prepare the inputs for the engine from the YAML file."""
    agro_management_inputs = test_data["AgroManagement"]
    cropd = test_data["ModelParameters"]

    weather_data_provider = WeatherDataProviderTestHelper(
        test_data["WeatherVariables"], meteo_range_checks=meteo_range_checks
    )
    crop_model_params_provider = ParameterProvider(cropdata=cropd)
    external_states = test_data["ExternalStates"]

    # convert parameters to tensors
    crop_model_params_provider.clear_override()
    for name in crop_model_params:
        value = torch.tensor(crop_model_params_provider[name], dtype=dtype)
        crop_model_params_provider.set_override(name, value, check=False)

    # convert external states to tensors
    tensor_external_states = [
        {k: v if k == "DAY" else torch.tensor(v, dtype=dtype) for k, v in item.items()}
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
    # the gradient for each table element separately
    # So, we flatten the parameter for easier indexing
    param_flat = param_value.reshape(-1)
    grad_flat = torch.zeros_like(param_flat)

    for i in range(param_flat.numel()):
        p_plus = param_flat.clone()
        p_plus[i] += delta
        p_minus = param_flat.clone()
        p_minus[i] -= delta

        p_plus = p_plus.view_as(param_value)
        p_minus = p_minus.view_as(param_value)

        model = get_model_fn()
        out_plus = model({param_name: p_plus})[out_name]
        loss_plus = out_plus.sum()

        model = get_model_fn()
        out_minus = model({param_name: p_minus})[out_name]
        loss_minus = out_minus.sum()

        grad_flat[i] = (loss_plus - loss_minus) / (2 * delta)

    return grad_flat.view_as(param_value)


class Afgen:
    """Differentiable AFGEN function, expanded from pcse.

    AFGEN is a linear interpolation function based on a table of XY pairs.
    Now supports batched tables (tensor of lists) for vectorized operations.
    """

    def _check_x_ascending(self, tbl_xy):
        """Checks that the x values are strictly ascending.

        Also truncates any trailing (0.,0.) pairs as a result of data coming
        from a CGMS database.

        Args:
            tbl_xy: Table of XY pairs as a tensor or array-like object.
                   Can be 1D (single table) or ND (vectorized tables).

        Returns:
            list or tensor: List of valid indices (for 1D) or tensor of valid counts (for ND).

        Raises:
            ValueError: If x values are not strictly ascending.
        """
        # Handle batched tables (>1D tensors)
        if tbl_xy.dim() > 1:
            batch_shape = tbl_xy.shape[:-1]
            table_len = tbl_xy.shape[-1]

            # Flatten batch dimensions for processing
            flat_tables = tbl_xy.reshape(-1, table_len)
            num_tables = flat_tables.shape[0]

            valid_counts = []
            for idx in range(num_tables):
                table = flat_tables[idx]
                x_list = table[0::2]
                y_list = table[1::2]
                n = len(x_list)

                # Find trailing (0, 0) pairs to truncate
                valid_n = n
                for i in range(n - 1, 0, -1):
                    if x_list[i] == 0 and y_list[i] == 0:
                        valid_n = i
                    else:
                        break

                # Check if x range is strictly ascending
                valid_x_list = x_list[:valid_n]
                for i in range(1, len(valid_x_list)):
                    if valid_x_list[i] <= valid_x_list[i - 1]:
                        msg = (
                            "X values for AFGEN input list"
                            + " not strictly ascending: {x_list.tolist()}"
                        )
                        raise ValueError(msg)

                valid_counts.append(valid_n)

            return torch.tensor(valid_counts).reshape(batch_shape)

        # Original 1D logic from pcse
        x_list = tbl_xy[0::2]
        y_list = tbl_xy[1::2]
        n = len(x_list)

        # Find trailing (0, 0) pairs to truncate
        valid_n = n
        for i in range(n - 1, 0, -1):
            if x_list[i] == 0 and y_list[i] == 0:
                valid_n = i
            else:
                break

        # Check only the valid (non-trailing-zero) portion
        valid_x_list = x_list[:valid_n]

        # Check if x range is strictly ascending
        for i in range(1, len(valid_x_list)):
            if valid_x_list[i] <= valid_x_list[i - 1]:
                msg = f"X values for AFGEN input list not strictly ascending: {x_list.tolist()}"
                raise ValueError(msg)

        return list(range(valid_n))

    def __init__(self, tbl_xy):
        # Convert to tensor if needed
        tbl_xy = torch.as_tensor(tbl_xy, dtype=DTYPE)

        # Detect if we have batched tables (>1D)
        self.is_batched = tbl_xy.dim() > 1

        if self.is_batched:
            self.batch_shape = tbl_xy.shape[:-1]
            table_len = tbl_xy.shape[-1]

            # Store the full batched tables
            self.tbl_xy = tbl_xy

            # Get valid counts for each table
            valid_counts = self._check_x_ascending(tbl_xy)
            self.valid_counts = valid_counts

            # Extract x and y for all tables
            flat_tables = tbl_xy.reshape(-1, table_len)
            num_tables = flat_tables.shape[0]

            x_list_batch = []
            y_list_batch = []
            slopes_batch = []

            for idx in range(num_tables):
                table = flat_tables[idx]
                valid_n = valid_counts.flatten()[idx].item()

                x_indices = torch.tensor([2 * i for i in range(valid_n)])
                y_indices = torch.tensor([2 * i + 1 for i in range(valid_n)])

                x_vals = table[x_indices]
                y_vals = table[y_indices]

                # Calculate slopes
                if len(x_vals) > 1:
                    slopes = (y_vals[1:] - y_vals[:-1]) / (x_vals[1:] - x_vals[:-1])
                else:
                    slopes = torch.tensor([], dtype=DTYPE)

                x_list_batch.append(x_vals)
                y_list_batch.append(y_vals)
                slopes_batch.append(slopes)

            # Store as lists - don't reshape, just keep the flat structure
            self.x_list_batch = x_list_batch
            self.y_list_batch = y_list_batch
            self.slopes_batch = slopes_batch

        else:
            # Original 1D logic from pcse
            self.batch_shape = None
            indices = self._check_x_ascending(tbl_xy)

            # Extract x and y values using indices
            x_indices = torch.tensor([2 * i for i in indices])
            y_indices = torch.tensor([2 * i + 1 for i in indices])
            self.x_list = tbl_xy[x_indices]
            self.y_list = tbl_xy[y_indices]

            # Calculate slopes
            x1 = self.x_list[:-1]
            x2 = self.x_list[1:]
            y1 = self.y_list[:-1]
            y2 = self.y_list[1:]
            self.slopes = (y2 - y1) / (x2 - x1)

    def __call__(self, x):
        """Returns the interpolated value at abscissa x.

        Args:
            x (torch.Tensor): The abscissa value at which to interpolate.
                             Can be scalar or batched to match table dimensions.

        Returns:
            torch.Tensor: The interpolated value, preserving batch dimensions.
        """
        x = torch.as_tensor(x, dtype=DTYPE)

        if self.is_batched:
            # Ensure x has compatible shape for broadcasting
            # x can be scalar or have batch dimensions

            # Flatten batch dimensions for processing
            flat_x = x.reshape(-1) if x.dim() > 0 else x.unsqueeze(0)
            num_queries = flat_x.shape[0] if flat_x.dim() > 0 else 1

            results = []

            # Process each table
            for idx in range(len(self.x_list_batch)):
                x_list = self.x_list_batch[idx]
                y_list = self.y_list_batch[idx]
                slopes = self.slopes_batch[idx]

                # Get the query value (broadcast if needed)
                if num_queries == 1:
                    x_val = flat_x[0] if flat_x.dim() > 0 else flat_x
                elif idx < num_queries:
                    x_val = flat_x[idx]
                else:
                    x_val = flat_x[0]  # Broadcast first value

                # Boundary conditions
                if x_val <= x_list[0]:
                    result = y_list[0]
                elif x_val >= x_list[-1]:
                    result = y_list[-1]
                else:
                    # Find interval and interpolate
                    i = torch.searchsorted(x_list, x_val, right=False) - 1
                    i = torch.clamp(i, 0, len(x_list) - 2)
                    result = y_list[i] + slopes[i] * (x_val - x_list[i])

                results.append(result)

            # Reshape to original batch shape
            output = torch.stack(results).reshape(self.batch_shape)
            return output

        # Original scalar logic from pcse
        # Clamp to boundaries
        if x <= self.x_list[0]:
            return self.y_list[0]
        if x >= self.x_list[-1]:
            return self.y_list[-1]

        # Find interval index using torch.searchsorted for differentiability
        i = torch.searchsorted(self.x_list, x, right=False) - 1
        i = torch.clamp(i, 0, len(self.x_list) - 2)

        # Linear interpolation
        v = self.y_list[i] + self.slopes[i] * (x - self.x_list[i])
        return v

    @property
    def shape(self):
        """Returns the shape of the Afgen table."""
        return self.batch_shape


class AfgenTrait(TraitType):
    """An AFGEN table trait.

    Attributes:
        default_value: Default Afgen instance with identity mapping.
        into_text: Description of the trait type.
    """

    default_value = Afgen([0, 0, 1, 1])
    into_text = "An AFGEN table of XY pairs"

    def validate(self, obj, value):
        """Validate that the value is an Afgen instance or an iterable to create one.

        Args:
            obj: The object instance containing this trait.
            value: The value to validate (either an Afgen instance or an iterable).

        Returns:
            Afgen: A validated Afgen instance.

        Raises:
            TraitError: If the value cannot be validated as an Afgen instance.
        """
        if isinstance(value, Afgen):
            return value
        elif isinstance(value, Iterable):
            return Afgen(value)
        self.error(obj, value)


def _get_params_shape(params):
    """Get the parameters shape.

    Parameters can have arbitrary number of dimensions, but all parameters that are not zero-
    dimensional should have the same shape.

    This check if fundamental for vectorized operations in the physical models.
    """
    shape = ()
    for parname in params.trait_names():
        # Skip special traitlets attributes
        if parname.startswith("trait"):
            continue
        param = getattr(params, parname)
        # Parameters that are not zero dimensional should all have the same shape
        if param.shape and not shape:
            shape = param.shape
        elif param.shape:
            assert param.shape == shape, (
                "All parameters should have the same shape (or have no dimensions)"
            )
    return shape


def _get_drv(drv_var, expected_shape):
    """Check that the driving variables have the expected shape and fetch them.

    Driving variables can be scalars (0-dimensional) or match the expected shape.
    Scalars will be broadcast during operations.

    [!] This function will be redundant once weathercontainer supports batched variables.

    Args:
        drv_var: driving variable in WeatherDataContainer
        expected_shape: Expected shape tuple for non-scalar variables

    Raises:
        ValueError: If any variable has incompatible shape

    Returns:
        torch.Tensor: The validated variable, either as-is or broadcasted to expected shape.
    """
    # Check shape: must be scalar (0-d) or match expected_shape
    if not isinstance(drv_var, torch.Tensor) or drv_var.dim() == 0:
        # Scalar is valid, will be broadcast
        return _broadcast_to(drv_var, expected_shape)
    elif drv_var.shape == expected_shape:
        # Matches expected shape
        return drv_var
    else:
        raise ValueError(
            f"Requested weather variable has incompatible shape {drv_var.shape}. "
            f"Expected scalar (0-dimensional) or shape {expected_shape}."
        )


def _broadcast_to(x, shape):
    """Create a view of tensor X with the given shape."""
    # If x is not a tensor, convert it
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=DTYPE)
    # If already the correct shape, return as-is
    if x.shape == shape:
        return x
    if x.dim() == 0:
        # For 0-d tensors, we simply broadcast to the given shape
        return torch.broadcast_to(x, shape)
    # The given shape should match x in all but the last axis, which represents
    # the dimension along which the time integration is carried out.
    # We first append an axis to x, then expand to the given shape
    return x.unsqueeze(-1).expand(shape)
