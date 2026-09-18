import datetime
import types
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from diffwofost.physical_models.weather import to_weather_data_iterator


class TestToWeatherDataIteratorFromDataFrame:
    def test_returns_iterator_of_weather_variables(self):
        weather_data = pd.DataFrame(
            {
                "DAY": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"],
                "TEMP": [10.0, 11.0, 9.0, 12.0],
            }
        )
        weather_data_iter = to_weather_data_iterator(weather_data)

        assert isinstance(weather_data_iter, types.GeneratorType)
        first = next(weather_data_iter)
        assert first["DAY"] == datetime.date(2020, 4, 1)
        assert torch.equal(first["TEMP"], torch.tensor(10.0))

    def test_raises_on_values_outside_valid_range(self):
        weather_data_faulty = pd.DataFrame(
            {
                "DAY": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"],
                "TEMP": [10.0, 1000.0, 9.0, 12.0],  # unrealistic temperature
            }
        )
        with pytest.raises(ValueError, match="outside the range"):
            to_weather_data_iterator(weather_data_faulty)

    def test_succeed_if_values_outside_valid_range_and_check_disabled(self):
        weather_data_faulty = pd.DataFrame(
            {
                "DAY": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"],
                "TEMP": [10.0, 1000.0, 9.0, 12.0],  # unrealistic temperature
            }
        )
        weather_data_iter = to_weather_data_iterator(weather_data_faulty, check=False)
        assert isinstance(weather_data_iter, types.GeneratorType)

    def test_raises_on_wrong_date_intervals(self):
        weather_data_faulty = pd.DataFrame(
            {
                "DAY": ["2020-04-01", "2020-04-03", "2020-04-04", "2020-04-05"],  # one day missing
                "TEMP": [10.0, 11.0, 9.0, 12.0],
            }
        )
        with pytest.raises(ValueError, match="consecutive daily dates"):
            to_weather_data_iterator(weather_data_faulty)

    def test_succeed_if_wrong_date_intervals_and_check_disabled(self):
        weather_data_faulty = pd.DataFrame(
            {
                "DAY": ["2020-04-01", "2020-04-03", "2020-04-04", "2020-04-05"],  # one day missing
                "TEMP": [10.0, 11.0, 9.0, 12.0],
            }
        )
        weather_data_iter = to_weather_data_iterator(weather_data_faulty, check=False)
        assert isinstance(weather_data_iter, types.GeneratorType)

    def test_nan_values_are_allowed_by_default(self):
        weather_data_with_nan = pd.DataFrame(
            {
                "DAY": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"],
                "TEMP": [10.0, 11.0, 9.0, np.nan],
            }
        )
        weather_data_iter = to_weather_data_iterator(weather_data_with_nan)
        assert isinstance(weather_data_iter, types.GeneratorType)

    def test_raises_on_nan_values_if_skipna(self):
        weather_data_with_nan = pd.DataFrame(
            {
                "DAY": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"],
                "TEMP": [10.0, 11.0, 9.0, np.nan],
            }
        )
        with pytest.raises(ValueError, match="NaN"):
            to_weather_data_iterator(weather_data_with_nan, skipna=False)

    def test_day_column_is_optional(self):
        weather_data = pd.DataFrame({"TEMP": [10.0, 11.0, 9.0, 12.0]})
        weather_data_iter = to_weather_data_iterator(weather_data)
        first = next(weather_data_iter)
        assert len(first) == 1
        assert "TEMP" in first


class TestToWeatherDataIteratorFromDataset:
    def test_raises_if_no_key_is_recognized(self):
        weather_data = xr.Dataset(
            {
                "MYTEMP": ("DAY", [10.0, 11.0, 9.0, 12.0]),
                "MYRAIN": ("DAY", [0.0, 1.0, 24.0, 1.0]),
            },
            coords={"DAY": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"]},
        )
        with pytest.raises(ValueError):
            to_weather_data_iterator(weather_data)

    def test_returns_iterator_of_weather_variables(self):
        weather_data = xr.Dataset(
            {"TEMP": ("DAY", [10.0, 11.0, 9.0, 12.0])},
            coords={"DAY": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"]},
        )
        weather_data_iter = to_weather_data_iterator(weather_data)

        assert isinstance(weather_data_iter, types.GeneratorType)
        first = next(weather_data_iter)
        assert first["DAY"] == datetime.date(2020, 4, 1)
        assert torch.equal(first["TEMP"], torch.tensor(10.0))

    def test_returns_iterator_inferring_dimension_names_from_day_coords(self):
        # the time dimension will be identified from the "DAY" variable or coordinate
        days = ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"]
        dim_name = "my_time"
        weather_data = xr.Dataset(
            {
                "TEMP": (
                    ("location", dim_name),
                    [
                        [10.0, 11.0, 9.0, 12.0],
                        [11.0, 7.0, 8.0, 13.0],
                        [9.0, 3.0, 11.0, 11.0],
                    ],
                )
            },
            coords={"DAY": (dim_name, days)},
        )
        weather_data_iter = to_weather_data_iterator(weather_data)
        assert isinstance(weather_data_iter, types.GeneratorType)

        weather_data_list = list(weather_data_iter)
        assert len(weather_data_list) == len(days)
        first = weather_data_list[0]
        assert first["DAY"] == datetime.date(2020, 4, 1)
        assert torch.equal(first["TEMP"], torch.tensor([10.0, 11.0, 9.0]))

    @pytest.mark.parametrize("dim_name", ["time", "day", "dates"])
    def test_returns_iterator_with_known_dimension_names(self, dim_name):
        # if the DAY variable/coordinate is missing, the time dimension can be recognized from some
        # default names
        weather_data = xr.Dataset(
            {
                "TEMP": (
                    ("location", dim_name),
                    [
                        [10.0, 11.0, 9.0, 12.0],
                        [11.0, 7.0, 8.0, 13.0],
                        [9.0, 3.0, 11.0, 11.0],
                    ],
                )
            },
        )
        weather_data_iter = to_weather_data_iterator(weather_data)

        assert isinstance(weather_data_iter, types.GeneratorType)
        weather_data_list = list(weather_data_iter)
        assert len(weather_data_list) == 4
        first = weather_data_list[0]
        assert torch.equal(first["TEMP"], torch.tensor([10.0, 11.0, 9.0]))

    def test_raises_if_time_dimension_not_recognized(self):
        # if the dimension name is not known and the "DAY" variable or coordinate is not present,
        # an error will be raised
        weather_data = xr.Dataset(
            {
                "TEMP": (
                    ("location", "my_time_dim"),
                    [
                        [10.0, 11.0, 9.0, 12.0],
                        [11.0, 7.0, 8.0, 13.0],
                        [9.0, 3.0, 11.0, 11.0],
                    ],
                )
            },
        )
        with pytest.raises(ValueError):
            to_weather_data_iterator(weather_data)

    def test_returns_iterator_if_time_dimension_specified_on_input(self):
        dim_name = "my_time_dim"
        weather_data = xr.Dataset(
            {
                "TEMP": (
                    ("location", dim_name),
                    [
                        [10.0, 11.0, 9.0, 12.0],
                        [11.0, 7.0, 8.0, 13.0],
                        [9.0, 3.0, 11.0, 11.0],
                    ],
                )
            },
        )
        weather_data_iter = to_weather_data_iterator(weather_data, time_dim=dim_name)
        assert isinstance(weather_data_iter, types.GeneratorType)

        weather_data_list = list(weather_data_iter)
        assert len(weather_data_list) == 4
        first = weather_data_list[0]
        assert torch.equal(first["TEMP"], torch.tensor([10.0, 11.0, 9.0]))

    def test_dataset_is_transposed_if_needed(self):
        dim_name = "my_time_dim"
        weather_data = xr.Dataset(
            {
                "TEMP": (
                    (dim_name, "location"),
                    [
                        [10.0, 11.0, 9.0, 12.0],
                        [11.0, 7.0, 8.0, 13.0],
                        [9.0, 3.0, 11.0, 11.0],
                    ],
                )
            },
        )
        weather_data_iter = to_weather_data_iterator(weather_data, time_dim=dim_name)
        assert isinstance(weather_data_iter, types.GeneratorType)

        weather_data_list = list(weather_data_iter)
        assert len(weather_data_list) == 3
        first = weather_data_list[0]
        assert torch.equal(first["TEMP"], torch.tensor([10.0, 11.0, 9.0, 12.0]))

    def test_raises_on_values_outside_valid_range(self):
        weather_data_faulty = xr.Dataset(
            {"TEMP": ("DAY", [10.0, 1000.0, 9.0, 12.0])},  # unrealistic temperature
            coords={"DAY": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"]},
        )
        with pytest.raises(ValueError, match="outside the range"):
            to_weather_data_iterator(weather_data_faulty)

    def test_raises_on_wrong_date_intervals(self):
        weather_data_faulty = xr.Dataset(
            {"TEMP": ("DAY", [10.0, 11.0, 9.0, 12.0])},
            coords={
                "DAY": ["2020-04-01", "2020-04-03", "2020-04-04", "2020-04-05"]
            },  # one day missing
        )
        with pytest.raises(ValueError, match="consecutive daily dates"):
            to_weather_data_iterator(weather_data_faulty)

    def test_raises_on_nan_values_if_skipna(self):
        weather_data_with_nan = xr.Dataset(
            {"TEMP": ("DAY", [10.0, 11.0, 9.0, np.nan])},
            coords={"DAY": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"]},
        )
        with pytest.raises(ValueError, match="NaN"):
            to_weather_data_iterator(weather_data_with_nan, skipna=False)

    def test_day_coordinate_is_optional(self):
        weather_data = xr.Dataset({"TEMP": ("DAY", [10.0, 11.0, 9.0, 12.0])})
        weather_data_iter = to_weather_data_iterator(weather_data)
        first = next(weather_data_iter)
        assert len(first) == 1
        assert "TEMP" in first
