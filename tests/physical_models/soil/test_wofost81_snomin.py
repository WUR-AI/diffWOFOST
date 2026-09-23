"""WOFOST 8.1 + SNOMIN matches PCSE daily, is differentiable, and batches."""

import datetime as dt
from pathlib import Path
import pytest
import torch
import yaml
from pcse.base import ParameterProvider as PcseParameterProvider
from pcse.base import WeatherDataContainer
from pcse.base import WeatherDataProvider
from pcse.input import YAMLAgroManagementReader
from pcse.input import YAMLCropDataProvider
from pcse.models import Wofost81_NWLP_MLWB_SNOMIN
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.wofost81 import Wofost81
from diffwofost.physical_models.engine import Engine
from diffwofost.physical_models.parameter_providers import ParameterProvider
from diffwofost.physical_models.soil.soil_wrappers import SoilModuleWrapper_NWLP_MLWB_SNOMIN
from diffwofost.physical_models.test import calculate_numerical_grad

_GYM = Path("/home/michiel/WUR/PCSE-Gym/pcse_gym/envs/configs")
_START = dt.date(2000, 3, 15)
_END = dt.date(2000, 4, 14)

# Daily crop, water and nitrogen outputs. Layered variables are compared per layer.
_DAILY = (
    "DVS",
    "LAI",
    "TAGP",
    "TWLV",
    "TWST",
    "TWSO",
    "TRA",
    "TRAMX",
    "NAVAIL",
    "NuptakeTotal",
    "NamountLV",
    "NamountST",
    "NamountRT",
    "NamountSO",
)
_LAYERED = ("SM", "NH4", "NO3")


def _weather_rows():
    day = _START
    rows = []
    while day <= _END:
        tmin, tmax = 8.0, 18.0
        temp = 0.5 * (tmin + tmax)
        rows.append(
            {
                "DAY": day,
                "LAT": 52.0,
                "LON": 5.67,
                "ELEV": 10.0,
                "IRRAD": 15.0e6,
                "TMIN": tmin,
                "TMAX": tmax,
                "TEMP": temp,
                "DTEMP": 0.5 * (temp + tmax),
                "VAP": 12.0,
                "RAIN": 0.2,
                "E0": 0.4,
                "ES0": 0.3,
                "ET0": 0.35,
                "WIND": 2.0,
            }
        )
        day += dt.timedelta(days=1)
    return rows


class _CallableWeather(WeatherDataProvider):
    def __init__(self, rows):
        super().__init__()
        for row in rows:
            container = WeatherDataContainer(**row)
            self._store_WeatherDataContainer(container, container.DAY)


def _providers():
    crop = YAMLCropDataProvider(Wofost81_NWLP_MLWB_SNOMIN, fpath=str(_GYM / "crop"))
    with (_GYM / "soil" / "arminda_soil.yaml").open() as handle:
        soil = yaml.safe_load(handle)
    with (_GYM / "site" / "arminda_site.yaml").open() as handle:
        site = yaml.safe_load(handle)
    agro_path = Path("/tmp/wofost81_snomin_agro.yaml")
    agro_path.write_text(
        """
AgroManagement:
- 2000-03-15:
    CropCalendar:
        crop_name: winterwheat
        variety_name: Arminda
        crop_start_date: 2000-03-15
        crop_start_type: sowing
        crop_end_date: 2000-08-01
        crop_end_type: harvest
        max_duration: 300
    TimedEvents: null
    StateEvents: null
"""
    )
    agro = YAMLAgroManagementReader(str(agro_path))
    return crop, soil, site, agro


def _config():
    return Configuration(
        CROP=Wofost81,
        SOIL=SoilModuleWrapper_NWLP_MLWB_SNOMIN,
        OUTPUT_VARS=[*_DAILY, *_LAYERED],
    )


def _numbers(value):
    """Flatten a scalar, array or tensor to a list of Python floats."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().reshape(-1)]
    if hasattr(value, "tolist"):
        raw = value.tolist()
        if isinstance(raw, list):
            flat = []
            for item in raw:
                if isinstance(item, list):
                    flat.extend(item)
                else:
                    flat.append(item)
            return [float(item) for item in flat]
        return [float(raw)]
    return [float(value)]


def _close(actual, expected, tol=1e-4):
    return abs(actual - expected) <= tol + tol * abs(expected)


def _assert_series(reference, model, tol=1e-4):
    assert len(reference) == len(model)
    names = (*_DAILY, *_LAYERED)
    for day_index, (ref_day, diff_day) in enumerate(zip(reference, model, strict=True)):
        assert ref_day["day"] == diff_day["day"]
        for name in names:
            ref_values = _numbers(ref_day.get(name))
            diff_values = _numbers(diff_day.get(name))
            if ref_values is None and diff_values is None:
                continue
            assert ref_values is not None and diff_values is not None, (day_index, name)
            assert len(ref_values) == len(diff_values), (day_index, name, ref_values, diff_values)
            for layer, (actual, expected) in enumerate(zip(diff_values, ref_values, strict=True)):
                assert _close(actual, expected, tol), (
                    name,
                    ref_day["day"],
                    layer,
                    actual,
                    expected,
                )


def _run_pcse(output_vars):
    crop, soil, site, agro = _providers()
    reference = Wofost81_NWLP_MLWB_SNOMIN(
        PcseParameterProvider(cropdata=crop, soildata=soil, sitedata=site),
        _CallableWeather(_weather_rows()),
        agro,
        output_vars=output_vars,
    )
    reference.run_till(_END)
    return reference.get_output()


def _run_diff(overrides=None):
    crop, soil, site, agro = _providers()
    params = ParameterProvider(cropdata=crop, soildata=soil, sitedata=site)
    params.set_active_crop("winterwheat", "Arminda", "sowing", "harvest")
    for name, value in (overrides or {}).items():
        params[name] = value
    model = Engine(_config())
    model.setup(params, iter(_weather_rows()), agro)
    model.run_till(_END)
    return model.get_output()


def _member(value, index):
    """Take one batch member from a stored output value."""
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value
        return value.select(-1, index)
    return value


@pytest.mark.skipif(not _GYM.exists(), reason="PCSE-Gym crop and soil files are not available")
def test_daily_outputs_match_pcse():
    """Every simulated day matches PCSE, not only the final day."""
    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    reference = _run_pcse([*_DAILY, *_LAYERED])
    model = _run_diff()
    _assert_series(reference, model)


class _LastDay:
    """Run one Arminda month and return the final-day outputs."""

    def __call__(self, overrides):
        last = _run_diff(overrides)[-1]
        result = {}
        for name, value in last.items():
            if isinstance(value, torch.Tensor):
                result[name] = value
            elif value is not None and name in {"TAGP", "SM", "NO3"}:
                result[name] = torch.tensor(value, dtype=torch.float64)
        return result


@pytest.mark.skipif(not _GYM.exists(), reason="PCSE-Gym crop and soil files are not available")
@pytest.mark.parametrize(
    ("parameter", "output_name"),
    [
        ("CVL", "TAGP"),
        ("WAV", "SM"),
        # Early-season uptake is demand-limited, so NuptakeTotal does not depend
        # on the nitrification rate. Nitrate does: KNIT_REF converts NH4 to NO3.
        ("KNIT_REF", "NO3"),
    ],
)
def test_autograd_matches_numerical_gradient(parameter, output_name):
    """Central differences agree with autograd for crop, water and SNOMIN parameters."""
    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    crop, soil, site, _agro = _providers()
    baseline = ParameterProvider(cropdata=crop, soildata=soil, sitedata=site)
    baseline.set_active_crop("winterwheat", "Arminda", "sowing", "harvest")
    param = torch.nn.Parameter(torch.tensor(float(baseline[parameter]), dtype=torch.float64))

    numerical_grad = calculate_numerical_grad(lambda: _LastDay(), parameter, param, output_name)
    loss = _LastDay()({parameter: param})[output_name].sum()
    autograd = torch.autograd.grad(loss, param)[0]

    assert torch.isfinite(autograd).all()
    assert torch.any(autograd != 0), (parameter, output_name, autograd)
    torch.testing.assert_close(numerical_grad, autograd, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not _GYM.exists(), reason="PCSE-Gym crop and soil files are not available")
def test_batched_trajectories_match_independent_runs():
    """Batch members keep separate emergence, water and nitrogen histories.

    TSUMEM changes the emergence day (minimum-temperature window and crop N).
    WAV changes the layered water balance. KNIT_REF changes SNOMIN nitrification.
    """
    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    crop, soil, site, _agro = _providers()
    baseline = ParameterProvider(cropdata=crop, soildata=soil, sitedata=site)
    baseline.set_active_crop("winterwheat", "Arminda", "sowing", "harvest")
    tsumem = float(baseline["TSUMEM"])
    wav = float(baseline["WAV"])
    knit = float(baseline["KNIT_REF"])
    member_values = {
        "TSUMEM": (tsumem, tsumem * 1.8),
        "WAV": (wav, wav * 0.25),
        "KNIT_REF": (knit, knit * 4.0),
    }
    batched = {
        name: torch.tensor(pair, dtype=torch.float64) for name, pair in member_values.items()
    }
    batch_output = _run_diff(batched)
    scalar_outputs = [
        _run_diff({name: torch.tensor(pair[index], dtype=torch.float64) for name, pair in member_values.items()})
        for index in range(2)
    ]

    names = ("DVS", "LAI", "TAGP", "TRA", "NAVAIL", "NuptakeTotal", "SM", "NH4", "NO3")
    for index, scalar in enumerate(scalar_outputs):
        assert len(scalar) == len(batch_output)
        for ref_day, batch_day in zip(scalar, batch_output, strict=True):
            assert ref_day["day"] == batch_day["day"]
            for name in names:
                expected = _numbers(ref_day[name])
                actual = _numbers(_member(batch_day[name], index))
                assert expected is not None and actual is not None
                for layer, (got, want) in enumerate(zip(actual, expected, strict=True)):
                    assert _close(got, want, tol=1e-5), (
                        index,
                        name,
                        ref_day["day"],
                        layer,
                        got,
                        want,
                    )

    last = batch_output[-1]
    dvs = [_numbers(_member(last["DVS"], index))[0] for index in range(2)]
    sm = [_numbers(_member(last["SM"], index))[0] for index in range(2)]
    navail = [_numbers(_member(last["NAVAIL"], index))[0] for index in range(2)]
    assert not _close(dvs[0], dvs[1], tol=1e-6)
    assert not _close(sm[0], sm[1], tol=1e-6)
    assert not _close(navail[0], navail[1], tol=1e-6)
