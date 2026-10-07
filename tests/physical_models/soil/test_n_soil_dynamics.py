"""Potential-production soil nitrogen against PCSE."""

import datetime
import torch
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.soil.n_soil_dynamics import N_PotentialProduction as PcseSupply
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.soil.n_soil_dynamics import N_PotentialProduction

DAY = datetime.date(2010, 6, 1)


def test_available_nitrogen_stays_at_the_potential_production_pool():
    """NAVAIL stays at 100 kg/ha after a day of integration, matching PCSE."""
    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    params = ParameterProvider(sitedata={})
    pcse = PcseSupply(DAY, VariableKiosk(), params)
    pcse.calc_rates(DAY, {})
    pcse.integrate(DAY)
    diff = N_PotentialProduction(DAY, VariableKiosk(), params)
    diff.calc_rates(DAY, {})
    diff.integrate(DAY)
    expected = torch.tensor(pcse.states.NAVAIL, dtype=diff.states.NAVAIL.dtype)
    torch.testing.assert_close(diff.states.NAVAIL, expected)
