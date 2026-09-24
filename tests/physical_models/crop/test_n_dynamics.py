"""Targeted check of WOFOST 8.1 crop nitrogen book-keeping against PCSE."""

import datetime
import torch
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.crop.n_dynamics import N_Crop_Dynamics as PcseNDynamics
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.crop.n_dynamics import N_Crop_Dynamics

DAY = datetime.date(2010, 6, 1)


def _kiosk(values):
    kiosk = VariableKiosk()
    for name, value in values.items():
        kiosk.register_variable(0, name, type="S", publish=True)
        kiosk.set_variable(0, name, value)
    return kiosk


def _parameters():
    return ParameterProvider(
        cropdata={
            "NMAXLV_TB": [0.0, 0.05, 2.0, 0.02],
            "DVS_N_TRANSL": 0.8,
            "NMAXRT_FR": 0.5,
            "NMAXST_FR": 0.5,
            "NMAXSO": 0.015,
            "TCNT": 10.0,
            "NFIX_FR": 0.0,
            "RNUPTAKEMAX": 10.0,
            "NRESIDLV": 0.004,
            "NRESIDST": 0.003,
            "NRESIDRT": 0.002,
        }
    )


def _states():
    return {
        "DVS": 1.0,
        "RFTRA": 1.0,
        "WLV": 1000.0,
        "WST": 800.0,
        "WRT": 400.0,
        "WSO": 500.0,
        "GRLV": 0.0,
        "GRST": 0.0,
        "GRRT": 0.0,
        "GRSO": 20.0,
        "NAVAIL": 50.0,
        "DRLV": 10.0,
        "DRST": 2.0,
        "DRRT": 1.0,
    }


def test_organ_nitrogen_loss_matches_pcse():
    """Initial organ nitrogen and the loss with dying biomass follow PCSE."""
    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    states = _states()
    pcse = PcseNDynamics(DAY, _kiosk(states), _parameters())
    pcse.calc_rates(DAY, {})
    tensors = {name: torch.tensor(value) for name, value in states.items()}
    diff = N_Crop_Dynamics(DAY, _kiosk(tensors), _parameters())
    diff.calc_rates(DAY, {})
    for name in ("NamountLV", "NamountST", "NamountRT", "RNdeathLV", "RNloss"):
        actual = getattr(diff.states if name.startswith("Namount") else diff.rates, name)
        expected = getattr(pcse.states if name.startswith("Namount") else pcse.rates, name)
        torch.testing.assert_close(
            actual, torch.tensor(expected, dtype=actual.dtype), rtol=1e-7, atol=1e-9
        )
