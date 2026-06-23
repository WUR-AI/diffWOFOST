from collections.abc import Iterator
from dataclasses import dataclass
import pandas as pd
import torch
from diffwofost.physical_models.config import ComputeConfig


@dataclass(frozen=True)
class WeatherVariable:
    unit: str
    min: float
    max: float


WEATHER_VARIABLES = dict(
    LAT=WeatherVariable("Degrees", -90.0, 90.0),
    LON=WeatherVariable("Degrees", -180.0, 180.0),
    ELEV=WeatherVariable("m", -300, 6000),
    IRRAD=WeatherVariable("J/m2/day", 0.0, 40e6),
    TMIN=WeatherVariable("Celsius", -50.0, 60.0),
    TMAX=WeatherVariable("Celsius", -50.0, 60.0),
    VAP=WeatherVariable("hPa", 0.06, 199.3),
    RAIN=WeatherVariable("cm/day", 0, 25),
    E0=WeatherVariable("cm/day", 0.0, 2.5),
    ES0=WeatherVariable("cm/day", 0.0, 2.5),
    ET0=WeatherVariable("cm/day", 0.0, 2.5),
    SNOWDEPTH=WeatherVariable("cm", 0.0, 250.0),
    TEMP=WeatherVariable("Celsius", -50.0, 60.0),
    TMINRA=WeatherVariable("Celsius", -50.0, 60.0),
    WIND=WeatherVariable("m/s", 0.0, 100.0),
)


def iterator_from_dataframe(df: pd.DataFrame, check: bool = True) -> Iterator:
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

    Yields:
        dict[str, typing.Any]: Weather variables as key-value pairs. Variables will be converted
            to torch tensors, using dtype and device as configured in `ComputeConfig`.
    """
    if "DAY" in df:
        days = pd.to_datetime(df["DAY"])
    else:
        days = None

    if check:
        # Check range of weather variables
        for var_name, var in WEATHER_VARIABLES.items():
            if var_name in df.columns:
                col = df[var_name]
                assert ((var.min <= col) & (col <= var.max)).all(), (
                    f"Values for `{var_name}` outside the range [{var.min}, {var.max}]"
                    f"(expected unit is {var.unit})."
                )

        # Check dates
        if days is not None:
            expected = pd.date_range(start=days.iloc[0], periods=len(days), freq="D")
            assert (days == expected).all(), (
                "Column `DAY` must contain consecutive daily dates with no gaps or duplicates."
            )

    device = ComputeConfig.get_device()
    dtype = ComputeConfig.get_dtype()

    vars = {
        var_name: torch.tensor(df[var_name].to_numpy(), device=device, dtype=dtype)
        for var_name in WEATHER_VARIABLES.keys()
        if var_name in df.columns
    }
    if days is not None:
        vars["DAY"] = days.dt.date.to_numpy()

    for n in range(len(df)):
        yield {k: v[n] for k, v in vars.items()}
