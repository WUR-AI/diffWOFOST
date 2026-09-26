"""Targeted check of WOFOST 8.1 nitrogen stress against the PCSE component."""

import datetime
import torch
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.crop.nutrients.n_stress import N_Stress as PcseNStress
from diffwofost.physical_models.crop.nutrients.n_stress import N_Stress

DAY = datetime.date(2010, 6, 1)


def _close(actual, expected):
    expected_tensor = torch.tensor(expected, dtype=actual.dtype, device=actual.device)
    assert torch.isclose(actual, expected_tensor, rtol=1e-7, atol=1e-9)


def _kiosk(values):
    kiosk = VariableKiosk()
    for name, value in values.items():
        kiosk.register_variable(0, name, type="S", publish=True)
        kiosk.set_variable(0, name, value)
    return kiosk


def _stress_parameters():
    return ParameterProvider(
        cropdata={
            "NMAXLV_TB": [0.0, 0.05, 2.0, 0.02],
            "NSLLV_TB": [0.0, 1.0, 1.0, 1.0, 2.0, 1.5],
            "NMAXRT_FR": 0.5,
            "NMAXST_FR": 0.5,
            "NRESIDLV": 0.004,
            "NRESIDST": 0.003,
            "NMAXSO": 0.015,
            "RGRLAI_MIN": 0.007,
            "RGRLAI": 0.008,
        }
    )


def _stress_states(n_amount_lv):
    return {
        "DVS": 1.0,
        "WLV": 1000.0,
        "WST": 1000.0,
        "WSO": 0.0,
        "NamountLV": n_amount_lv,
        "NamountST": 20.0,
        "NamountSO": 0.0,
    }


def test_n_stress_matches_pcse_at_two_leaf_concentrations():
    """NSLLV and RFRGRL follow PCSE both below and above the critical leaf nitrogen."""
    for n_amount_lv in (10.0, 40.0):
        states = _stress_states(n_amount_lv)
        pcse = PcseNStress(DAY, _kiosk(states), _stress_parameters())
        pcse(DAY, {})
        tensor_states = {name: torch.tensor(value) for name, value in states.items()}
        diff = N_Stress(DAY, _kiosk(tensor_states), _stress_parameters())
        diff.calc_rates(DAY, {})
        _close(diff.rates.NSLLV, pcse.rates.NSLLV)
        _close(diff.rates.RFRGRL, pcse.rates.RFRGRL)
