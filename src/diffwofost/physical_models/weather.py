from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import torch
import xarray as xr
from diffwofost.physical_models.config import ComputeConfig


@dataclass(frozen=True)
class WeatherVariable:
    unit: str
    min: float
    max: float


# These are the weather variables recognized by diffWOFOST internally, along
# with their units and valid ranges.
WEATHER_VARIABLES = {
    "LAT": WeatherVariable("Degrees", -90.0, 90.0),
    "LON": WeatherVariable("Degrees", -180.0, 180.0),
    "ELEV": WeatherVariable("m", -300, 6000),
    "IRRAD": WeatherVariable("J/m2/day", 0.0, 40e6),
    "TMIN": WeatherVariable("Celsius", -50.0, 60.0),
    "TMAX": WeatherVariable("Celsius", -50.0, 60.0),
    "VAP": WeatherVariable("hPa", 0.06, 199.3),
    "RAIN": WeatherVariable("cm/day", 0, 25),
    "E0": WeatherVariable("cm/day", 0.0, 2.5),
    "ES0": WeatherVariable("cm/day", 0.0, 2.5),
    "ET0": WeatherVariable("cm/day", 0.0, 2.5),
    "SNOWDEPTH": WeatherVariable("cm", 0.0, 250.0),
    "TEMP": WeatherVariable("Celsius", -50.0, 60.0),
    "TMINRA": WeatherVariable("Celsius", -50.0, 60.0),
    "WIND": WeatherVariable("m/s", 0.0, 100.0),
    "DTEMP": WeatherVariable("Celsius", -50.0, 60.0),
}

TIME_DIM_NAMES = {"day", "time", "dates"}


def to_weather_data_iterator(
    data: pd.DataFrame | xr.Dataset,
    check: bool = True,
    skipna: bool = True,
    time_dim: str | None = None,
) -> Iterator:
    """Weather data generator from a Pandas DataFrame or an xarray Dataset.

    This utility function transforms weather data from tabular or nd-array format to an iterator of
    torch tensors that can be fed to diffWOFOST's engine.

    Args:
        data (pd.DataFrame | xr.Dataset): DataFrame or Dataset containing weather data. Weather
            variables should be listed as columns (DataFrame) or data variables (Dataset). In order
            to be interpreted as weather variables, they should be named as the keys of
            `diffwofost.physical_models.weather.WEATHER_VARIABLES`. Rows/elements are expected to
            represent daily time steps (an optional column/1D-coordinate named "DAY" should list the
            corresponding dates).
        check (bool, optional): Optionally carry out validity checks for the dataset. Defaults to
            True.
        skipna (bool, optional): How to handle NaN values when `check` is True. If True, allow NaN
            values as part of the weather data.
        time_dim (str, optional): name of the dimension to iterate over. Only relevant if `data` is
            a xr.Dataset object. If not provided, the function will try to guess it from:
            * the name of the dimension of the "DAY" coordinate (if present).
            * the first element of `diffwofost.physical_models.weather.TIME_DIM_NAMES` in
                `data.dims`.

    Yields:
        dict[str, typing.Any]: Weather variables as key-value pairs. Variables will be converted
            to torch tensors, using dtype and device as configured in `ComputeConfig`.

    Examples:
        >>> import pandas as pd
        >>> weather_data = pd.DataFrame({
        ...     "DAY": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"],
        ...     "TEMP": [10., 11., 9., 12.],
        ... })
        >>> weather_data_iter = iterator_from_dataframe(weather_data)
        >>> next(weather_data_iter)
        {'DAY': datetime.date(2020, 4, 1), 'TEMP': tensor(10.)}
        >>> weather_data_faulty = pd.DataFrame({
        ...     "DAY": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"],
        ...     "TEMP": [10., 1000., 9., 12.], # unrealistic temperature
        ... })
        >>> iterator_from_dataframe(weather_data_faulty)
        ValueError: Values for `TEMP` outside the range [-50.0, 60.0] (expected unit is Celsius).

    """
    _check_weather_variable_keys(data)

    iter_dim = _get_iterator_dimension(data, time_dim)
    iter_length = _get_iterator_length(data, iter_dim)

    dates = _extract_dates_if_present(data)

    if check:
        _check_weather_variables_range(data, skipna=skipna)
        if dates is not None:
            _check_dates(dates)

    variables = {}

    # if present, include the dates to the returned variables
    if dates is not None:
        variables["DAY"] = dates

    variables.update(_to_dict_of_tensors(data, iter_dim))

    return _iterate(variables, length=iter_length)


def _check_weather_variable_keys(data: pd.DataFrame | xr.Dataset) -> None:
    num_variables = len([var_name for var_name in WEATHER_VARIABLES if var_name in data])
    if num_variables < 1:
        raise ValueError(
            "No weather variable found. Variables should be named as the keys of "
            "`diffwofost.physical_models.weather.WEATHER_VARIABLES`."
        )


def _get_iterator_dimension(
    data: pd.DataFrame | xr.Dataset, time_dim: str | None = None
) -> str | None:
    """Determine the dimension to iterate over.

    This is only relevant for xr.Dataset objects whose variables have >= 2 dimensions: if the
    weather variables are 1D, the only available dimension will be used for iteration.
    """
    if isinstance(data, pd.DataFrame) or _are_all_weather_variables_less_than_2d(data):
        # there is only one dimension to iterate over.
        return None
    elif time_dim:
        # if the time dimension is provided, only check if it's a valid dimension name
        assert time_dim in data.dims, f"Dimension {time_dim} missing from dimensions {data.dims}."
        return time_dim
    elif "DAY" in data:
        # if the time dimension is not provided, check the dimension of the "DAY" coordinate
        # (if present)
        day = data["DAY"]
        assert day.ndim == 1, "Daily dates should be provided as a 1D-coordinate."
        return day.dims[0]
    else:
        # if none of the above, check sensible names for the time dimension
        for guess in TIME_DIM_NAMES:
            if guess in data.dims:
                return guess
    raise ValueError(
        f"Cannot determine which dimension to iterate over for dataset with dims: {data.dims}."
    )


def _get_iterator_length(data: pd.DataFrame | xr.Dataset, iter_dim: str | None) -> int:
    return len(data[iter_dim]) if iter_dim is not None else len(data)


def _are_all_weather_variables_less_than_2d(data: xr.Dataset) -> bool:
    return all(
        [var.ndim < 2 for var_name, var in data.variables.items() if var_name in WEATHER_VARIABLES]
    )


def _extract_dates_if_present(data: pd.DataFrame | xr.Dataset) -> np.ndarray | None:
    if "DAY" not in data:
        return None
    else:
        day = data["DAY"].values
        return pd.to_datetime(day).date


def _check_weather_variables_range(data: pd.DataFrame | xr.Dataset, skipna: bool = True) -> None:
    for var_name, var_range in WEATHER_VARIABLES.items():
        if var_name in data:
            var = data[var_name]
            is_null = var.isnull()
            if not skipna and is_null.any():
                raise ValueError(f"{var_name} includes {int(is_null.sum())} NaN values.")
            outside_range = (var < var_range.min) | (var > var_range.max)
            is_invalid = outside_range.where(~is_null, other=False)
            if is_invalid.any():
                raise ValueError(
                    f"Values for `{var_name}` outside the range [{var_range.min}, {var_range.max}] "
                    f"(expected unit is {var_range.unit})."
                )


def _check_dates(dates: np.ndarray) -> None:
    expected = pd.date_range(start=dates[0], periods=len(dates), freq="D")
    if not (dates == expected).all():
        raise ValueError(
            "Column `DAY` must contain consecutive daily dates with no gaps or duplicates."
        )


def _to_dict_of_tensors(
    data: pd.DataFrame | xr.Dataset,
    iter_dim: str | None = None,
) -> dict[str, torch.Tensor]:
    return {
        var_name: _to_tensor(data[var_name], iter_dim)
        for var_name in WEATHER_VARIABLES
        if var_name in data
    }


def _to_tensor(data: pd.Series | xr.DataArray, iter_dim: str | None = None) -> torch.Tensor:
    device = ComputeConfig.get_device()
    dtype = ComputeConfig.get_dtype()
    if isinstance(data, xr.DataArray) and iter_dim is not None:
        data = data.transpose(iter_dim, ...)
    return torch.tensor(data.to_numpy(), device=device, dtype=dtype)


def _iterate(variables: dict[str, Any], length):
    for n in range(length):
        yield {k: v[n] for k, v in variables.items()}
