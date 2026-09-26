"""Layered water balance against PCSE before crop emergence."""

import datetime
from types import SimpleNamespace
import torch
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.input import WOFOST81SiteDataProvider_SNOMIN
from pcse.soil.multilayer_waterbalance import WaterBalanceLayered as PcseWater
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.soil.multilayer_waterbalance import WaterBalanceLayered
from .test_wofost81_snomin import _example_soil

DAY = datetime.date(2010, 6, 1)


def _parameters():
    soil = _example_soil()
    n_layers = len(soil["SoilProfileDescription"]["SoilLayers"])
    site = WOFOST81SiteDataProvider_SNOMIN(
        WAV=20.0,
        CO2=360.0,
        NH4I=[10.0] * n_layers,
        NO3I=[1.0] * n_layers,
    )
    return ParameterProvider(soildata=soil, sitedata=site)


def test_initial_moisture_and_one_rain_day_match_pcse():
    """Initial layer moisture and the first rainy day's flow follow PCSE."""
    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    params = _parameters()
    weather = SimpleNamespace(RAIN=5.0, E0=0.4, ES0=0.3)
    pcse = PcseWater(DAY, VariableKiosk(), params)
    pcse.calc_rates(DAY, weather)
    diff = WaterBalanceLayered(DAY, VariableKiosk(), params)
    diff.calc_rates(DAY, weather)
    torch.testing.assert_close(
        diff.states.SM, torch.tensor(pcse.states.SM, dtype=torch.float64), rtol=1e-6, atol=1e-8
    )
    torch.testing.assert_close(
        diff.states.WC, torch.tensor(pcse.states.WC, dtype=torch.float64), rtol=1e-6, atol=1e-8
    )
    torch.testing.assert_close(
        diff.rates.EVS, torch.tensor(pcse.rates.EVS, dtype=torch.float64), rtol=1e-6, atol=1e-8
    )
    torch.testing.assert_close(
        diff.rates.Flow, torch.tensor(pcse.rates.Flow, dtype=torch.float64), rtol=1e-6, atol=1e-8
    )
