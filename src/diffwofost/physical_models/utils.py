"""This file contains code that is required to run the YAML unit tests.

It contains:
    - EngineTestHelper: engine specifically for running the YAML tests.
    - WeatherDataProviderTestHelper: a weatherdata provides that takes the weather
      inputs from the YAML file.

Note that the code here is *not* python2 compatible.
"""

import logging
import math
from collections import namedtuple
from collections.abc import Iterable
import torch
import yaml
from pcse import signals
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.weather import WeatherDataContainer
from pcse.base.weather import WeatherDataProvider
from pcse.settings import settings
from pcse.traitlets import TraitType
from pcse.util import doy
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.engine import Engine

logging.disable(logging.CRITICAL)


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
            weather_inputs = {k: v for k, v in weather.items() if k != "SNOWDEPTH"}
            wdc = WeatherDataContainer(**weather_inputs)
            self._store_WeatherDataContainer(wdc, wdc.DAY)


def prepare_engine_input(
    test_data, crop_model_params, device=None, dtype=None, meteo_range_checks=True
):
    """Prepare the inputs for the engine from the YAML file."""
    # If not specified, use default dtype and device
    if device is None:
        device = ComputeConfig.get_device()
    if dtype is None:
        dtype = ComputeConfig.get_dtype()

    agro_management_inputs = test_data["AgroManagement"]
    cropd = test_data["ModelParameters"]

    weather_data_provider = WeatherDataProviderTestHelper(
        test_data["WeatherVariables"], meteo_range_checks=meteo_range_checks
    )

    # The PCSE WeatherDataContainer stores required variables as Python floats.
    # Some of our tests rely on weather inputs being torch.Tensors (e.g. to
    # broadcast/batch weather variables). We only do this conversion when
    # METEO_RANGE_CHECKS is disabled because the PCSE range checks assume
    # scalar floats.
    if not meteo_range_checks:
        for (_, _), wdc in weather_data_provider.store.items():
            for varname in (
                "IRRAD",
                "TMIN",
                "TMAX",
                "TEMP",
                "VAP",
                "RAIN",
                "WIND",
                "E0",
                "ES0",
                "ET0",
            ):
                if hasattr(wdc, varname):
                    value = getattr(wdc, varname)
                    if not isinstance(value, torch.Tensor):
                        setattr(wdc, varname, torch.tensor(value, dtype=dtype, device=device))
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


def daylength(day, latitude, angle=-4, dtype=None, device=None):
    """PyTorch-vectorized daylength calculation for a given day, latitude and base angle.

    Derived from the WOFOST routine ASTRO.FOR and simplified to include only
    daylength calculation. When ``angle == -4`` (the default) the result is
    identical to the ``DAYLP`` field returned by :func:`astro`.

    Args:
        day (datetime.date): the day for which to calculate daylength.
        latitude (float or torch.Tensor): latitude of location (scalar or torch.Tensor)
        angle (float): The photoperiodic daylength starts/ends when the sun
            is `angle` degrees under the horizon. Default is -4 degrees.
        dtype (torch.dtype): torch dtype to use (defaults to ComputeConfig.get_dtype())
        device (torch.device): torch device to use (defaults to ComputeConfig.get_device())

    Returns:
        torch.Tensor: daylength for the given day and latitude.
    """
    if dtype is None:
        dtype = ComputeConfig.get_dtype()
    if device is None:
        device = ComputeConfig.get_device()

    # Convert latitude to tensor so all ops are vectorized and differentiable.
    if not isinstance(latitude, torch.Tensor):
        latitude = torch.tensor(latitude, dtype=dtype, device=device)

    # Check for range of latitude
    if (latitude.abs() > 90.0).any():
        msg = "Latitude not between -90 and 90"
        raise RuntimeError(msg)

    # Calculate day-of-year from date object day
    IDAY = doy(day)

    # calculate daylength
    # Declination only depends on IDAY so it stays a Python scalar for efficiency.
    DEC = -math.asin(
        math.sin(23.45 * math.radians(1.0)) * math.cos(2.0 * math.pi * (float(IDAY) + 10.0) / 365.0)
    )
    SINLD = torch.sin(math.radians(1.0) * latitude) * math.sin(DEC)
    COSLD = torch.cos(math.radians(1.0) * latitude) * math.cos(DEC)
    AOB = (-math.sin(angle * math.radians(1.0)) + SINLD) / COSLD

    # daylength — replace scalar if/elif/else with torch.where for batched support
    aob_clamped = AOB.clamp(-1.0, 1.0)
    DAYLP_base = 12.0 * (1.0 + 2.0 * torch.asin(aob_clamped) / math.pi)
    DAYLP = torch.where(
        AOB > 1.0,
        torch.full_like(AOB, 24.0),
        torch.where(AOB < -1.0, torch.zeros_like(AOB), DAYLP_base),
    )

    return DAYLP


# Named tuple for returning results of ASTRO
astro_nt = namedtuple("AstroResults", "DAYL, DAYLP, SINLD, COSLD, DIFPP, ATMTR, DSINBE, ANGOT")


def astro(day, latitude, radiation, dtype=None, device=None):
    """PyTorch-vectorized version of the ASTRO routine.

    This subroutine calculates astronomic daylength, diurnal radiation
    characteristics such as the atmospheric transmission, diffuse radiation etc.
    Inputs `latitude` and `radiation` can be Python scalars or torch.Tensors,
    enabling fully batched, differentiable computation.

    output is a `namedtuple` in the following order and tags::

        DAYL      Astronomical daylength (base = 0 degrees)     h
        DAYLP     Astronomical daylength (base =-4 degrees)     h
        SINLD     Seasonal offset of sine of solar height       -
        COSLD     Amplitude of sine of solar height             -
        DIFPP     Diffuse irradiation perpendicular to
                  direction of light                         J m-2 s-1
        ATMTR     Daily atmospheric transmission                -
        DSINBE    Daily total of effective solar height         s
        ANGOT     Angot radiation at top of atmosphere       J m-2 d-1

    Args:
        day (datetime.date): the day for which to calculate astronomic daylength.
        latitude (float or torch.Tensor): latitude of location
        radiation (float or torch.Tensor): daily global incoming radiation in J/m2/day
        dtype (torch.dtype): torch dtype to use (defaults to ComputeConfig.get_dtype())
        device (torch.device): torch device to use (defaults to ComputeConfig.get_device())

    Returns:
        a named tuple containing the calculated astronomic daylength and related variables.
    """
    if dtype is None:
        dtype = ComputeConfig.get_dtype()
    if device is None:
        device = ComputeConfig.get_device()

    # Convert latitude and radiation to tensors so all downstream ops are
    # fully differentiable and support arbitrary batch shapes.
    if not isinstance(latitude, torch.Tensor):
        latitude = torch.tensor(latitude, dtype=dtype, device=device)
    if not isinstance(radiation, torch.Tensor):
        radiation = torch.tensor(radiation, dtype=dtype, device=device)

    # Check for range of latitude
    if (latitude.abs() > 90.0).any():
        msg = "Latitude not between -90 and 90"
        raise RuntimeError(msg)

    # Determine day-of-year (IDAY) from day
    IDAY = doy(day)

    # Declination and solar constant for this day
    # DEC and SC only depend on IDAY so remain Python scalars for efficiency.
    DEC = -math.asin(
        math.sin(23.45 * math.radians(1.0)) * math.cos(2.0 * math.pi * (float(IDAY) + 10.0) / 365.0)
    )
    SC = 1370.0 * (1.0 + 0.033 * math.cos(2.0 * math.pi * float(IDAY) / 365.0))

    # calculation of daylength from intermediate variables
    # SINLD, COSLD and AOB
    SINLD = torch.sin(math.radians(1.0) * latitude) * math.sin(DEC)
    COSLD = torch.cos(math.radians(1.0) * latitude) * math.cos(DEC)
    AOB = SINLD / COSLD

    # For very high latitudes and days in summer and winter a limit is
    # inserted to avoid math errors when daylength reaches 24 hours in
    # summer or 0 hours in winter.

    # Calculate solution for base=0 degrees
    # Clamp AOB to [-1, 1] before asin to guard against floating-point overflow.
    aob_clamped = AOB.clamp(-1.0, 1.0)
    sqrt_term = torch.sqrt(torch.clamp(1.0 - aob_clamped**2, min=0.0))
    DAYL_base = 12.0 * (1.0 + 2.0 * torch.asin(aob_clamped) / math.pi)
    DAYL = torch.where(
        AOB > 1.0,
        torch.full_like(AOB, 24.0),
        torch.where(AOB < -1.0, torch.zeros_like(AOB), DAYL_base),
    )
    # integrals of sine of solar height
    DSINB = torch.where(
        AOB.abs() <= 1.0,
        3600.0 * (DAYL * SINLD + 24.0 * COSLD * sqrt_term / math.pi),
        3600.0 * (DAYL * SINLD),
    )
    DSINBE = torch.where(
        AOB.abs() <= 1.0,
        3600.0
        * (
            DAYL * (SINLD + 0.4 * (SINLD**2 + COSLD**2 * 0.5))
            + 12.0 * COSLD * (2.0 + 3.0 * 0.4 * SINLD) * sqrt_term / math.pi
        ),
        3600.0 * (DAYL * (SINLD + 0.4 * (SINLD**2 + COSLD**2 * 0.5))),
    )

    # Calculate solution for base=-4 degrees
    AOB_CORR = (-math.sin(math.radians(-4.0)) + SINLD) / COSLD
    aob_corr_clamped = AOB_CORR.clamp(-1.0, 1.0)
    DAYLP_base = 12.0 * (1.0 + 2.0 * torch.asin(aob_corr_clamped) / math.pi)
    DAYLP = torch.where(
        AOB_CORR > 1.0,
        torch.full_like(AOB_CORR, 24.0),
        torch.where(AOB_CORR < -1.0, torch.zeros_like(AOB_CORR), DAYLP_base),
    )

    # extraterrestrial radiation and atmospheric transmission
    ANGOT = SC * DSINB
    # Check for DAYL=0 as in that case the angot radiation is 0 as well
    ATMTR = torch.where(DAYL > 0.0, radiation / ANGOT, torch.zeros_like(radiation))

    # estimate fraction diffuse irradiation
    FRDIF = torch.where(
        ATMTR > 0.75,
        torch.full_like(ATMTR, 0.23),
        torch.where(
            (ATMTR <= 0.75) & (ATMTR > 0.35),
            1.33 - 1.46 * ATMTR,
            torch.where(
                (ATMTR <= 0.35) & (ATMTR > 0.07),
                1.0 - 2.3 * (ATMTR - 0.07) ** 2,
                torch.ones_like(ATMTR),  # ATMTR <= 0.07
            ),
        ),
    )

    DIFPP = FRDIF * ATMTR * 0.5 * SC

    return astro_nt(DAYL, DAYLP, SINLD, COSLD, DIFPP, ATMTR, DSINBE, ANGOT)


class Afgen:
    """Differentiable AFGEN function, expanded from pcse.

    AFGEN is a linear interpolation function based on a table of XY pairs.
    Now supports batched tables (tensor of lists) for vectorized operations.
    """

    @property
    def device(self):
        """Get device from ComputeConfig."""
        from diffwofost.physical_models.config import ComputeConfig

        return ComputeConfig.get_device()

    @property
    def dtype(self):
        """Get dtype from ComputeConfig."""
        from diffwofost.physical_models.config import ComputeConfig

        return ComputeConfig.get_dtype()

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
        tbl_xy = torch.as_tensor(tbl_xy, dtype=self.dtype, device=self.device)
        # If the table was provided as ints, promote to float so interpolation
        # doesn't truncate query points (e.g. 2.5 -> 2) and autograd works.
        if not tbl_xy.is_floating_point():
            tbl_xy = tbl_xy.to(dtype=self.dtype)

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
            flat_valid = valid_counts.reshape(-1).to(device=self.device)
            num_tables = flat_tables.shape[0]
            max_n = int(flat_valid.max().item()) if num_tables > 0 else 0

            # Store padded tensors so we can vectorize __call__.
            pad_x = torch.finfo(tbl_xy.dtype).max
            x_flat = torch.full((num_tables, max_n), pad_x, dtype=self.dtype, device=self.device)
            y_flat = torch.zeros((num_tables, max_n), dtype=self.dtype, device=self.device)
            slopes_flat = torch.zeros(
                (num_tables, max(0, max_n - 1)), dtype=self.dtype, device=self.device
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
                self.slopes = torch.tensor([], dtype=self.dtype, device=self.device)

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
            # Ensure contiguous query tensor to avoid internal copies in searchsorted
            x_query = x_vals.unsqueeze(1).contiguous()
            i = torch.searchsorted(self._x_flat, x_query, right=False) - 1
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


def _broadcast_to(x, shape, dtype=None, device=None):
    """Create a view of tensor X with the given shape.

    Args:
        x: The tensor or value to broadcast
        shape: The target shape
        dtype: Optional dtype for the tensor
        device: Optional device for the tensor
    """
    # Make sure x is a tensor
    x = torch.as_tensor(x, dtype=dtype)
    if device is not None:
        x = x.to(device=device)
    # If already the correct shape, return as-is
    if x.shape == shape:
        return x
    return torch.broadcast_to(x, shape)


def _snapshot_state(obj):
    return {name: val.clone() for name, val in obj.__dict__.items() if torch.is_tensor(val)}


def _restore_state(obj, snapshot):
    for name, val in snapshot.items():
        setattr(obj, name, val)


def _afgen_y_mask(table_1d: torch.Tensor) -> torch.Tensor:
    """Mask selecting the Y entries in a flattened AFGEN XY table.

    AFGEN XY tables are commonly stored as a flat vector `[x0, y0, x1, y1, ...]`
    with optional trailing `(0,0)` pairs as padding. This mask selects only the
    Y entries of the *valid* (unpadded) part to avoid turning trailing `(0,0)`
    into `(0, delta)` when perturbing parameters.
    """
    x_list = table_1d[0::2]
    y_list = table_1d[1::2]

    # Match the Afgen validation logic: truncate trailing (0,0) pairs, but if the
    # entire table is (0,0), keep the first pair.
    nonzero = ~(x_list.eq(0) & y_list.eq(0))
    last_valid = int(nonzero.nonzero()[-1].item()) if bool(nonzero.any()) else 0
    valid_n = last_valid + 1

    mask = torch.zeros_like(table_1d)
    mask[1 : 2 * valid_n : 2] = 1
    return mask
