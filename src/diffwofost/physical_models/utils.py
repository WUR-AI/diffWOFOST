"""This file contains code that is required to run the YAML unit tests.

It contains:
    - VariableKioskTestHelper: A subclass of the VariableKiosk that can use externally
      forced states/rates
    - EngineTestHelper: engine specifically for running the YAML tests.
    - WeatherDataProviderTestHelper: a weatherdata provides that takes the weather
      inputs from the YAML file.

Note that the code here is *not* python2 compatible.
"""

import logging
from collections.abc import Iterable
from pathlib import Path
import torch
import yaml
from pcse import signals
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.base.weather import WeatherDataContainer
from pcse.base.weather import WeatherDataProvider
from pcse.engine import BaseEngine
from pcse.settings import settings
from pcse.timer import Timer
from pcse.traitlets import Enum
from pcse.traitlets import TraitType
from .config import Configuration
from .engine import Engine

logging.disable(logging.CRITICAL)


class VariableKioskTestHelper(VariableKiosk):
    """Variable Kiosk for testing purposes which allows to use external states."""

    external_state_list = None

    def __init__(self, external_state_list=None):
        super().__init__()
        self.current_externals = {}
        if external_state_list:
            self.external_state_list = external_state_list

    def __call__(self, day):
        """Sets the external state/rate variables for the current day.

        Returns True if the list of external state/rate variables is exhausted,
        otherwise False.
        """
        if self.external_state_list:
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


class EngineTestHelper(Engine):
    """An engine which is purely for running the YAML unit tests."""

    def __init__(
        self,
        parameterprovider,
        weatherdataprovider,
        agromanagement,
        config,
        external_states=None,
        device=None,
        dtype=None,
    ):
        BaseEngine.__init__(self)

        # If a path is given, load the model configuration from a PCSE config file
        if isinstance(config, str | Path):
            self.mconf = Configuration.from_pcse_config_file(config)
        else:
            self.mconf = config

        self.parameterprovider = parameterprovider

        # Configure device and dtype on crop module class if it supports them
        if hasattr(self.mconf.CROP, "device") and device is not None:
            self.mconf.CROP.device = device
        if hasattr(self.mconf.CROP, "dtype") and dtype is not None:
            self.mconf.CROP.dtype = dtype

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
    test_data, crop_model_params, meteo_range_checks=True, dtype=torch.float64, device="cpu"
):
    """Prepare the inputs for the engine from the YAML file."""
    agro_management_inputs = test_data["AgroManagement"]
    cropd = test_data["ModelParameters"]

    weather_data_provider = WeatherDataProviderTestHelper(
        test_data["WeatherVariables"], meteo_range_checks=meteo_range_checks
    )
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

        def _valid_n_and_check(x_list: torch.Tensor, y_list: torch.Tensor) -> int:
            # Truncate trailing (0,0) pairs. If all pairs are (0,0), keep first pair.
            nonzero = ~(x_list.eq(0) & y_list.eq(0))
            last_valid = int(nonzero.nonzero()[-1].item()) if bool(nonzero.any()) else 0
            valid_n = last_valid + 1

            x_valid = x_list[:valid_n]
            if x_valid.numel() > 1 and not bool(torch.all(torch.diff(x_valid) > 0)):
                raise ValueError(
                    f"X values for AFGEN input list not strictly ascending: {x_list.tolist()}"
                )
            return valid_n

        if tbl_xy.dim() > 1:
            batch_shape = tbl_xy.shape[:-1]
            table_len = tbl_xy.shape[-1]
            flat = tbl_xy.reshape(-1, table_len)
            counts = [_valid_n_and_check(t[0::2], t[1::2]) for t in flat]
            return torch.tensor(counts, device=tbl_xy.device).reshape(batch_shape)

        valid_n = _valid_n_and_check(tbl_xy[0::2], tbl_xy[1::2])
        return list(range(valid_n))

    def __init__(self, tbl_xy):
        # Convert to tensor if needed
        tbl_xy = torch.as_tensor(tbl_xy)
        # If the table was provided as ints, promote to float so interpolation
        # doesn't truncate query points (e.g. 2.5 -> 2) and autograd works.
        if not tbl_xy.is_floating_point():
            tbl_xy = tbl_xy.to(dtype=torch.get_default_dtype())

        # Detect if we have batched tables (>1D)
        self.is_batched = tbl_xy.dim() > 1

        if self.is_batched:
            self.batch_shape = tbl_xy.shape[:-1]
            table_len = tbl_xy.shape[-1]

            # Keep the full batched tables for debugging/inspection
            self.tbl_xy = tbl_xy

            # Validate and compute how many (x,y) pairs are valid per table
            valid_counts = self._check_x_ascending(tbl_xy)
            self.valid_counts = valid_counts

            flat_tables = tbl_xy.reshape(-1, table_len)
            flat_valid = valid_counts.reshape(-1).to(device=tbl_xy.device)
            num_tables = flat_tables.shape[0]
            max_n = int(flat_valid.max().item()) if num_tables > 0 else 0

            # Store padded tensors so we can vectorize __call__.
            pad_x = torch.finfo(tbl_xy.dtype).max
            x_flat = torch.full(
                (num_tables, max_n), pad_x, dtype=tbl_xy.dtype, device=tbl_xy.device
            )
            y_flat = torch.zeros((num_tables, max_n), dtype=tbl_xy.dtype, device=tbl_xy.device)
            slopes_flat = torch.zeros(
                (num_tables, max(0, max_n - 1)), dtype=tbl_xy.dtype, device=tbl_xy.device
            )

            for idx in range(num_tables):
                n = int(flat_valid[idx].item())
                table = flat_tables[idx]
                x_vals = table[0::2][:n]
                y_vals = table[1::2][:n]

                x_flat[idx, :n] = x_vals
                y_flat[idx, :n] = y_vals
                if n < max_n:
                    y_flat[idx, n:] = y_vals[-1]
                if n > 1:
                    slopes_flat[idx, : n - 1] = (y_vals[1:] - y_vals[:-1]) / (
                        x_vals[1:] - x_vals[:-1]
                    )

            self._x_flat = x_flat
            self._y_flat = y_flat
            self._slopes_flat = slopes_flat
            self._valid_counts_flat = flat_valid

        else:
            # Original 1D logic from pcse
            self.batch_shape = None
            indices = self._check_x_ascending(tbl_xy)
            valid_n = len(indices)

            self.x_list = tbl_xy[0::2][:valid_n]
            self.y_list = tbl_xy[1::2][:valid_n]
            if valid_n > 1:
                self.slopes = (self.y_list[1:] - self.y_list[:-1]) / (
                    self.x_list[1:] - self.x_list[:-1]
                )
            else:
                self.slopes = torch.tensor([], dtype=tbl_xy.dtype, device=tbl_xy.device)

    def __call__(self, x):
        """Returns the interpolated value at abscissa x.

        Args:
            x (torch.Tensor): The abscissa value at which to interpolate.
                             Can be scalar or batched to match table dimensions.

        Returns:
            torch.Tensor: The interpolated value, preserving batch dimensions.
        """
        if self.is_batched:
            x = torch.as_tensor(x, dtype=self._x_flat.dtype, device=self._x_flat.device)
            flat_x = x.reshape(-1) if x.dim() > 0 else x.unsqueeze(0)
            num_tables = self._x_flat.shape[0]

            if flat_x.numel() == 1:
                x_vals = flat_x.expand(num_tables)
            elif flat_x.numel() == num_tables:
                x_vals = flat_x
            else:
                x_vals = flat_x[0].expand(num_tables)

            # Find interval index per table
            i = torch.searchsorted(self._x_flat, x_vals.unsqueeze(1), right=False) - 1
            i = i.squeeze(1)
            upper = torch.clamp(self._valid_counts_flat - 2, min=0)
            i = torch.clamp(i, min=0)
            i = torch.minimum(i, upper)

            idx = i.unsqueeze(1)
            x_i = self._x_flat.gather(1, idx).squeeze(1)
            y_i = self._y_flat.gather(1, idx).squeeze(1)
            slope_i = self._slopes_flat.gather(1, idx).squeeze(1)
            interp = y_i + slope_i * (x_vals - x_i)

            x0 = self._x_flat[:, 0]
            y0 = self._y_flat[:, 0]
            last_idx = (self._valid_counts_flat - 1).to(dtype=torch.long).unsqueeze(1)
            x_last = self._x_flat.gather(1, last_idx).squeeze(1)
            y_last = self._y_flat.gather(1, last_idx).squeeze(1)

            out = torch.where(
                x_vals <= x0,
                y0,
                torch.where(x_vals >= x_last, y_last, interp),
            )
            return out.reshape(self.batch_shape)

        x = torch.as_tensor(x, dtype=self.x_list.dtype, device=self.x_list.device)

        # Ensure contiguous memory layout for searchsorted
        x_list_contig = self.x_list.contiguous()
        x_contig = x.contiguous() if isinstance(x, torch.Tensor) and x.dim() > 0 else x

        # Find interval index using torch.searchsorted for differentiability
        i = torch.searchsorted(x_list_contig, x_contig, right=False) - 1
        i = torch.clamp(i, 0, len(self.x_list) - 2)

        # Calculate interpolated value
        interp_value = self.y_list[i] + self.slopes[i] * (x - self.x_list[i])

        # Apply boundary conditions using torch.where
        result = torch.where(
            x <= self.x_list[0],
            self.y_list[0],
            torch.where(x >= self.x_list[-1], self.y_list[-1], interp_value),
        )

        return result

    def to(self, device=None, dtype=None):
        """Move internal tensors to a different device/dtype (PyTorch-style).

        This is an in-place operation and returns ``self`` for chaining.
        """
        if device is None and dtype is None:
            return self

        for name in (
            "tbl_xy",
            "x_list",
            "y_list",
            "slopes",
            "_x_flat",
            "_y_flat",
            "_slopes_flat",
            "valid_counts",
            "_valid_counts_flat",
        ):
            if not hasattr(self, name):
                continue
            t = getattr(self, name)
            if not isinstance(t, torch.Tensor):
                continue
            # Keep integer tensors as integers; only move device for them.
            if t.is_floating_point():
                setattr(self, name, t.to(device=device, dtype=dtype))
            else:
                setattr(self, name, t.to(device=device))

        return self

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
        # Skip Enum and str parameters
        if isinstance(param, Enum) or isinstance(param, str):
            continue
        # Parameters that are not zero dimensional should all have the same shape
        if param.shape and not shape:
            shape = param.shape
        elif param.shape:
            assert param.shape == shape, (
                "All parameters should have the same shape (or have no dimensions)"
            )
    return shape


def _get_drv(drv_var, expected_shape, dtype, device=None):
    """Check that the driving variables have the expected shape and fetch them.

    Driving variables can be scalars (0-dimensional) or match the expected shape.
    Scalars will be broadcast during operations.

    [!] This function will be redundant once weathercontainer supports batched variables.

    Args:
        drv_var: driving variable in WeatherDataContainer
        expected_shape: Expected shape tuple for non-scalar variables
        dtype: dtype for the tensor
        device: Optional device for the tensor

    Raises:
        ValueError: If any variable has incompatible shape

    Returns:
        torch.Tensor: The validated variable, either as-is or broadcasted to expected shape.
    """
    # Check shape: must be scalar (0-d) or match expected_shape
    if not isinstance(drv_var, torch.Tensor) or drv_var.dim() == 0:
        # Scalar is valid, will be broadcast
        return _broadcast_to(drv_var, expected_shape, dtype, device)
    elif drv_var.shape == expected_shape:
        # Matches expected shape
        if dtype is not None:
            drv_var = drv_var.to(dtype=dtype)
        if device is not None:
            drv_var = drv_var.to(device=device)
        return drv_var
    else:
        raise ValueError(
            f"Requested weather variable has incompatible shape {drv_var.shape}. "
            f"Expected scalar (0-dimensional) or shape {expected_shape}."
        )


def _broadcast_to(x, shape, dtype, device=None):
    """Create a view of tensor X with the given shape.

    Args:
        x: The tensor or value to broadcast
        shape: The target shape
        dtype: dtype for the tensor
        device: Optional device for the tensor
    """
    # If x is not a tensor, convert it
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype)
    # Ensure correct dtype and device
    if dtype is not None:
        x = x.to(dtype=dtype)
    if device is not None:
        x = x.to(device=device)
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


def _snapshot_state(obj):
    return {name: val.clone() for name, val in obj.__dict__.items() if torch.is_tensor(val)}


def _restore_state(obj, snapshot):
    for name, val in snapshot.items():
        setattr(obj, name, val)
