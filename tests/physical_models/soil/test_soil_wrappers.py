"""The two ported WOFOST 8.1 soil wrappers against PCSE."""

import datetime
from types import SimpleNamespace
import torch
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.soil.soil_wrappers import SoilModuleWrapper_NWLP_MLWB_SNOMIN as PcseSnomin
from pcse.soil.soil_wrappers import SoilModuleWrapper_PP as PcsePP
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.soil.soil_wrappers import SoilModuleWrapper_NWLP_MLWB_SNOMIN
from diffwofost.physical_models.soil.soil_wrappers import SoilModuleWrapper_PP

DAY = datetime.date(2010, 6, 1)


def test_potential_production_wrapper_matches_pcse():
    """Field-capacity moisture and the constant nitrogen pool follow PCSE."""
    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    params = ParameterProvider(soildata={"SMFCF": 0.25})
    weather = SimpleNamespace(ES0=0.3, E0=0.4, RAIN=0.0)
    pcse = PcsePP(DAY, VariableKiosk(), params)
    pcse.calc_rates(DAY, weather)
    pcse.integrate(DAY)
    diff = SoilModuleWrapper_PP(DAY, VariableKiosk(), params)
    diff.calc_rates(
        DAY,
        {
            "ES0": torch.tensor(0.3),
            "E0": torch.tensor(0.4),
            "RAIN": torch.tensor(0.0),
        },
    )
    diff.integrate(DAY)
    sm = diff.waterbalance.states.SM
    navail = diff.nutrientbalance.states.NAVAIL
    torch.testing.assert_close(sm, torch.tensor(pcse.waterbalance.states.SM, dtype=sm.dtype))
    torch.testing.assert_close(
        navail, torch.tensor(pcse.nutrientbalance.states.NAVAIL, dtype=navail.dtype)
    )


def test_ported_wrappers_pair_the_same_balances_as_pcse():
    """Each wrapper points at the same water and nitrogen classes as PCSE."""
    assert SoilModuleWrapper_PP.waterbalance_class.__name__ == PcsePP.waterbalance_class.__name__
    assert (
        SoilModuleWrapper_PP.nutrientbalance_class.__name__ == PcsePP.nutrientbalance_class.__name__
    )
    assert (
        SoilModuleWrapper_NWLP_MLWB_SNOMIN.waterbalance_class.__name__
        == PcseSnomin.waterbalance_class.__name__
    )
    assert (
        SoilModuleWrapper_NWLP_MLWB_SNOMIN.nutrientbalance_class.__name__
        == PcseSnomin.nutrientbalance_class.__name__
    )
