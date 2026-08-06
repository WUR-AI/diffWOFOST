from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
import pandas as pd
import torch
from diffwofost.physical_models.config import ComputeConfig


@dataclass(frozen=True)
class WeatherVariable:
    unit: str
    min: float
    max: float


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


def iterator_from_dataframe(df: pd.DataFrame, check: bool = True, skipna: bool = True) -> Iterator:
    """Weather data generator from a Pandas DataFrame.

    This utility function transforms weather data from tabular format to an iterator of torch
    tensors that can be fed to diffWOFOST's engine.

    Args:
        df (pd.DataFrame): DataFrame containing weather data. Weather variables should be listed
            along columns. In order to be interpreted as weather variables, columns should be named
            as the keys of `diffwofost.physical_models.weather.WEATHER_VARIABLES`. Rows are expected
            to represent daily time steps (an optional column named "DAY" should list the
            corresponding dates).
        check (bool, optional): Optionally carry out validity checks for the dataset. Defaults to
            True.
        skipna (bool, optional): How to handle NaN values when `check` is True. If True, allow NaN
            values as part of the weather data.

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
    dates = _extract_dates_if_present(df)

    if check:
        _check_range_of_weather_variables(df, skipna=skipna)
        if dates is not None:
            _check_dates(dates)

    variables = {}

    # if dates are present, add them to the returned variables, converting them to datetime objects
    if dates is not None:
        variables["DAY"] = dates.dt.date.to_numpy()

    variables.update(_to_dict_of_tensors(df))

    return _iterate(variables, length=len(df))


def _extract_dates_if_present(df: pd.DataFrame) -> pd.Series | None:
    return pd.to_datetime(df["DAY"]) if "DAY" in df else None


def _check_range_of_weather_variables(df: pd.DataFrame, skipna: bool = True) -> None:
    for var_name, var in WEATHER_VARIABLES.items():
        if var_name in df.columns:
            col = df[var_name]
            is_nan = col.isna()
            if skipna:
                col = col[~is_nan]
            else:
                if is_nan.any():
                    raise ValueError(f"{var_name} includes {is_nan.sum()} NaN values.")
            if ((col < var.min) | (col > var.max)).any():
                raise ValueError(
                    f"Values for `{var_name}` outside the range [{var.min}, {var.max}] "
                    f"(expected unit is {var.unit})."
                )


def _check_dates(dates: pd.Series) -> None:
    expected = pd.date_range(start=dates.iloc[0], periods=len(dates), freq="D")
    if not (dates == expected).all():
        raise ValueError(
            "Column `DAY` must contain consecutive daily dates with no gaps or duplicates."
        )


def _to_dict_of_tensors(df: pd.DataFrame) -> dict[str, torch.Tensor]:
    device = ComputeConfig.get_device()
    dtype = ComputeConfig.get_dtype()
    return {
        var_name: torch.tensor(df[var_name].to_numpy(), device=device, dtype=dtype)
        for var_name in WEATHER_VARIABLES.keys()
        if var_name in df.columns
    }


def _iterate(variables: dict[str, Any], length):
    for n in range(length):
        yield {k: v[n] for k, v in variables.items()}
