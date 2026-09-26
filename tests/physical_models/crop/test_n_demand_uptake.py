"""Targeted check of WOFOST 8.1 nitrogen demand and translocation against PCSE."""

import datetime
import torch
from pcse.base.parameter_providers import ParameterProvider
from pcse.base.variablekiosk import VariableKiosk
from pcse.crop.nutrients.n_demand_uptake import N_Demand_Uptake as PcseNDemand
from diffwofost.physical_models.crop.nutrients.n_demand_uptake import N_Demand_Uptake

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


def _uptake_parameters():
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


def _uptake_states(dvs):
    return {
        "DVS": dvs,
        "RFTRA": 1.0,
        "WLV": 1000.0,
        "WST": 800.0,
        "WRT": 400.0,
        "WSO": 500.0,
        "NamountLV": 40.0,
        "NamountST": 15.0,
        "NamountRT": 8.0,
        "NamountSO": 2.0,
        "GRLV": 0.0,
        "GRST": 0.0,
        "GRRT": 0.0,
        "GRSO": 20.0,
        "NAVAIL": 50.0,
    }


def test_translocation_matches_pcse_before_and_after_threshold():
    """Checks N translocation against PCSE before and after DVS_N_TRANSL."""
    for dvs in (0.5, 1.2):
        states = _uptake_states(dvs)
        pcse = PcseNDemand(DAY, _kiosk(states), _uptake_parameters())
        pcse.calc_rates(DAY, {})
        tensor_states = {name: torch.tensor(value) for name, value in states.items()}
        diff = N_Demand_Uptake(DAY, _kiosk(tensor_states), _uptake_parameters())
        diff.calc_rates(DAY, {})
        if dvs < 0.8:
            _close(diff.rates.RNtranslocation, 0.0)
        _close(diff.rates.RNtranslocation, pcse.rates.RNtranslocation)
        _close(diff.rates.RNuptake, pcse.rates.RNuptake)


def _rate_after_threshold(overrides, rate_name):
    """One translocation day, with ``overrides`` replacing crop parameters."""
    from diffwofost.physical_models.config import ComputeConfig

    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    states = _uptake_states(1.2)
    tensors = {name: torch.tensor(value, dtype=torch.float64) for name, value in states.items()}
    provider = _uptake_parameters()
    for name, value in overrides.items():
        provider.set_override(name, value, check=False)
    model = N_Demand_Uptake(DAY, _kiosk(tensors), provider)
    model.calc_rates(DAY, {})
    return getattr(model.rates, rate_name)


def _assert_scalar_gradient(rate_name, parameter, value):
    param = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))
    loss = _rate_after_threshold({parameter: param}, rate_name).sum()
    autograd = torch.autograd.grad(loss, param)[0]
    delta = 1e-6
    with torch.no_grad():
        plus = _rate_after_threshold({parameter: param.detach() + delta}, rate_name).sum()
        minus = _rate_after_threshold({parameter: param.detach() - delta}, rate_name).sum()
    numerical = (plus - minus) / (2 * delta)
    assert torch.isfinite(autograd).all()
    assert autograd.item() != 0
    torch.testing.assert_close(autograd, numerical, rtol=1e-3, atol=1e-3)


def test_tcnt_gradient_matches_numerical():
    """TCNT changes translocation once development has passed DVS_N_TRANSL."""
    _assert_scalar_gradient("RNtranslocation", "TCNT", 10.0)


def test_nmaxso_gradient_matches_numerical():
    """NMAXSO changes storage-organ nitrogen demand after the translocation threshold."""
    _assert_scalar_gradient("NdemandSO", "NMAXSO", 0.015)
