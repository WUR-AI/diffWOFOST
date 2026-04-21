from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.utils import _broadcast_to


@dataclass(frozen=True)
class WeatherVariable:
    unit: str
    range: tuple[float, float]


# Units and ranges for meteorological variables
WEATHER_VARIABLES = {
    "LAT": WeatherVariable("Degrees", (-90.0, 90.0)),
    "LON": WeatherVariable("Degrees", (-180.0, 180.0)),
    "ELEV": WeatherVariable("m", (-300, 6000)),
    "IRRAD": WeatherVariable("J/m2/day", (0.0, 40e6)),
    "TMIN": WeatherVariable("Celsius", (-50.0, 60.0)),
    "TMAX": WeatherVariable("Celsius", (-50.0, 60.0)),
    "VAP": WeatherVariable(
        "hPa", (0.06, 199.3)
    ),  # computed as sat. vapour pressure at -50, 60 Celsius  # noqa: E501
    "RAIN": WeatherVariable("cm/day", (0, 25)),
    "E0": WeatherVariable("cm/day", (0.0, 2.5)),
    "ES0": WeatherVariable("cm/day", (0.0, 2.5)),
    "ET0": WeatherVariable("cm/day", (0.0, 2.5)),
    "SNOWDEPTH": WeatherVariable("cm", (0.0, 250.0)),
    "TEMP": WeatherVariable("Celsius", (-50.0, 60.0)),
    "TMINRA": WeatherVariable("Celsius", (-50.0, 60.0)),
    "WIND": WeatherVariable("m/s", (0.0, 100.0)),
}


class TensorWeatherDataProvider(ABC):
    def __init__(self, store, meteo_range_checks=True):
        self._shape = None

        # Optionally carry out range checks for the weather variables
        if meteo_range_checks:
            self._meteo_range_check(store)

        # Extract weather from store
        self.store = store
        self.shape = self._get_variable_shape()

    @abstractmethod
    def _get_variable_shape(self):
        """Determine the shape of the weather variables, excluding the time dimension.

        Returns:
            tuple: Base shape of the weather variables.
        """
        pass

    @abstractmethod
    def _get_variables(self, day):
        """Extract the available weather variables for the given date from the store.

        Args:
            day (datetime.date): date

        Returns:
            dict: Collection of available variables as extracted from the store.
        """
        pass

    def __call__(self, day):
        """Get the available weather variables on a given date.

        Args:
            day (datetime.date): date

        Returns:
            dict: Collection of available weather variables as `torch.Tensor`
        """
        vars = {}
        for key, var in self._get_variables(day).items():
            vars[key] = _broadcast_to(
                var,
                shape=self.shape,
                dtype=ComputeConfig.get_dtype(),
                device=ComputeConfig.get_device(),
            )
        return vars

    @property
    def shape(self):
        """Base shape of the weather variables.

        This is the shape of the data variables excluding the time dimension.
        """
        return self._shape

    @shape.setter
    def shape(self, shape):
        if self.shape and self.shape != shape:
            raise ValueError(f"Container shape already set to {self.shape}")
        self._shape = shape

    def _meteo_range_check(self, store):
        """Check whether entries in the store fit acceptable ranges."""
        raise NotImplementedError("Range checks have to be implemented.")
