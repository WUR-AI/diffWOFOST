import datetime
import types
import numpy as np
import pandas as pd
import pytest
import torch
from diffwofost.physical_models.weather import iterator_from_dataframe


class TestIteratorFromDataFrame:
    def test_returns_iterator_of_weather_variables(self):
        weather_data = pd.DataFrame(
            {
                "DAY": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"],
                "TEMP": [10.0, 11.0, 9.0, 12.0],
            }
        )
        weather_data_iter = iterator_from_dataframe(weather_data)

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
            iterator_from_dataframe(weather_data_faulty)

    def test_succeed_if_values_outside_valid_range_and_check_disabled(self):
        weather_data_faulty = pd.DataFrame(
            {
                "DAY": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"],
                "TEMP": [10.0, 1000.0, 9.0, 12.0],  # unrealistic temperature
            }
        )
        weather_data_iter = iterator_from_dataframe(weather_data_faulty, check=False)
        assert isinstance(weather_data_iter, types.GeneratorType)

    def test_raises_on_wrong_date_intervals(self):
        weather_data_faulty = pd.DataFrame(
            {
                "DAY": ["2020-04-01", "2020-04-03", "2020-04-04", "2020-04-05"],  # one day missing
                "TEMP": [10.0, 11.0, 9.0, 12.0],
            }
        )
        with pytest.raises(ValueError, match="consecutive daily dates"):
            iterator_from_dataframe(weather_data_faulty)

    def test_succeed_if_wrong_date_intervals_and_check_disabled(self):
        weather_data_faulty = pd.DataFrame(
            {
                "DAY": ["2020-04-01", "2020-04-03", "2020-04-04", "2020-04-05"],  # one day missing
                "TEMP": [10.0, 11.0, 9.0, 12.0],
            }
        )
        weather_data_iter = iterator_from_dataframe(weather_data_faulty, check=False)
        assert isinstance(weather_data_iter, types.GeneratorType)

    def test_nan_values_are_allowed_by_default(self):
        weather_data_with_nan = pd.DataFrame(
            {
                "DAY": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"],
                "TEMP": [10.0, 11.0, 9.0, np.nan],
            }
        )
        weather_data_iter = iterator_from_dataframe(weather_data_with_nan)
        assert isinstance(weather_data_iter, types.GeneratorType)

    def test_raises_on_nan_values_if_skipna(self):
        weather_data_with_nan = pd.DataFrame(
            {
                "DAY": ["2020-04-01", "2020-04-02", "2020-04-03", "2020-04-04"],
                "TEMP": [10.0, 11.0, 9.0, np.nan],
            }
        )
        with pytest.raises(ValueError, match="NaN"):
            iterator_from_dataframe(weather_data_with_nan, skipna=False)

    def test_day_column_is_optional(self):
        weather_data = pd.DataFrame({"TEMP": [10.0, 11.0, 9.0, 12.0]})
        weather_data_iter = iterator_from_dataframe(weather_data)
        first = next(weather_data_iter)
        assert len(first) == 1
        assert "TEMP" in first
