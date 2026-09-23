"""WOFOST 8.1 + SNOMIN matches PCSE daily, is differentiable, and batches.

Crop parameters come from the WOFOST 8.1 repository that PCSE selects for this
model. The soil profile is the example in ``pcse.soil.soil_profile.SoilProfile``.
Site values are the defaults of ``WOFOST81SiteDataProvider_SNOMIN``, plus the
four quantities that provider requires. Weather is the first month of a PCSE
7.2 test series.
"""

import datetime as dt
import inspect
import textwrap
from functools import lru_cache
from pathlib import Path
import pytest
import torch
import yaml
from pcse.base import ParameterProvider as PcseParameterProvider
from pcse.base import WeatherDataContainer
from pcse.base import WeatherDataProvider
from pcse.input import WOFOST81SiteDataProvider_SNOMIN
from pcse.input import YAMLAgroManagementReader
from pcse.input import YAMLCropDataProvider
from pcse.models import Wofost81_NWLP_MLWB_SNOMIN
from pcse.soil.soil_profile import SoilProfile
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.wofost81 import Wofost81
from diffwofost.physical_models.engine import Engine
from diffwofost.physical_models.parameter_providers import ParameterProvider
from diffwofost.physical_models.soil.soil_wrappers import SoilModuleWrapper_NWLP_MLWB_SNOMIN
from diffwofost.physical_models.test import calculate_numerical_grad
from diffwofost.physical_models.test import get_test_data

_WEATHER = (
    Path(__file__).resolve().parents[1] / "test_data" / "test_potentialproduction_wofost72_05.yaml"
)
_N_DAYS = 31
_CROP_NAME = "wheat"
_VARIETY_NAME = "Winter_wheat_101"

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


@lru_cache(maxsize=1)
def _weather_rows():
    """First month of weather from a PCSE potential-production test file."""
    rows = []
    for row in get_test_data(_WEATHER)["WeatherVariables"][:_N_DAYS]:
        item = {key: value for key, value in row.items() if key != "SNOWDEPTH"}
        item["DTEMP"] = 0.5 * (item["TEMP"] + item["TMAX"])
        rows.append(item)
    return rows


def _period():
    rows = _weather_rows()
    return rows[0]["DAY"], rows[-1]["DAY"]


class _CallableWeather(WeatherDataProvider):
    def __init__(self, rows):
        super().__init__()
        for row in rows:
            container = WeatherDataContainer(**row)
            self._store_WeatherDataContainer(container, container.DAY)


def _example_soil():
    """Load the soil-profile example published in the PCSE SoilProfile docstring."""
    lines = inspect.getdoc(SoilProfile).splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "SoilLayerTypes:")
    end = next(i for i, line in enumerate(lines) if line.strip().startswith("GroundWater:"))
    parsed = yaml.safe_load(textwrap.dedent("\n".join(lines[start : end + 1])))
    profile = parsed["SoilProfileDescription"]
    # The example's SubSoilType merge omits FSOMI. SoilLayer always reads it,
    # and the same example sets FSOMI to 0 on the deeper layers.
    if "FSOMI" not in profile["SubSoilType"]:
        profile["SubSoilType"]["FSOMI"] = 0.0
    rootable = sum(layer["Thickness"] for layer in profile["SoilLayers"])
    return {"RDMSOL": float(rootable), "SoilProfileDescription": profile}


def _providers():
    crop = YAMLCropDataProvider(Wofost81_NWLP_MLWB_SNOMIN)
    soil = _example_soil()
    n_layers = len(soil["SoilProfileDescription"]["SoilLayers"])
    # WAV, CO2, NH4I and NO3I have no default. The amounts sit inside the
    # ranges checked by WOFOST81SiteDataProvider_SNOMIN.
    site = WOFOST81SiteDataProvider_SNOMIN(
        WAV=20.0,
        CO2=360.0,
        NH4I=[10.0] * n_layers,
        NO3I=[1.0] * n_layers,
    )
    start, _end = _period()
    harvest = start + dt.timedelta(days=300)
    agro_path = Path("/tmp/wofost81_snomin_agro.yaml")
    agro_path.write_text(
        f"""
AgroManagement:
- {start.isoformat()}:
    CropCalendar:
        crop_name: {_CROP_NAME}
        variety_name: {_VARIETY_NAME}
        crop_start_date: {start.isoformat()}
        crop_start_type: sowing
        crop_end_date: {harvest.isoformat()}
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


def _run_pcse(output_vars, site_overrides=None):
    crop, soil, site, agro = _providers()
    if site_overrides:
        site = {**site, **site_overrides}
    reference = Wofost81_NWLP_MLWB_SNOMIN(
        PcseParameterProvider(cropdata=crop, soildata=soil, sitedata=site),
        _CallableWeather(_weather_rows()),
        agro,
        output_vars=output_vars,
    )
    _start, end = _period()
    reference.run_till(end)
    return reference.get_output()


def _run_diff(overrides=None):
    crop, soil, site, agro = _providers()
    params = ParameterProvider(cropdata=crop, soildata=soil, sitedata=site)
    params.set_active_crop(_CROP_NAME, _VARIETY_NAME, "sowing", "harvest")
    for name, value in (overrides or {}).items():
        params[name] = value
    model = Engine(_config())
    model.setup(params, iter(_weather_rows()), agro)
    _start, end = _period()
    model.run_till(end)
    return model.get_output()


def _member(value, index):
    """Take one batch member from a stored output value."""
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value
        return value.select(-1, index)
    return value


def test_daily_outputs_match_pcse():
    """Every simulated day matches PCSE, not only the final day."""
    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    reference = _run_pcse([*_DAILY, *_LAYERED])
    model = _run_diff()
    _assert_series(reference, model)


class _LastDay:
    """Run one month and return the final-day outputs."""

    def __call__(self, overrides):
        last = _run_diff(overrides)[-1]
        result = {}
        for name, value in last.items():
            if isinstance(value, torch.Tensor):
                result[name] = value
            elif value is not None and name in {"TAGP", "SM", "NO3"}:
                result[name] = torch.tensor(value, dtype=torch.float64)
        return result


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
    baseline.set_active_crop(_CROP_NAME, _VARIETY_NAME, "sowing", "harvest")
    param = torch.nn.Parameter(torch.tensor(float(baseline[parameter]), dtype=torch.float64))

    numerical_grad = calculate_numerical_grad(lambda: _LastDay(), parameter, param, output_name)
    loss = _LastDay()({parameter: param})[output_name].sum()
    autograd = torch.autograd.grad(loss, param)[0]

    assert torch.isfinite(autograd).all()
    assert torch.any(autograd != 0), (parameter, output_name, autograd)
    torch.testing.assert_close(numerical_grad, autograd, rtol=1e-3, atol=1e-3)


def test_batched_trajectories_match_independent_runs():
    """Batch members keep separate emergence, water and nitrogen histories.

    TSUMEM changes the emergence day (minimum-temperature window and crop N).
    WAV changes the layered water balance. KNIT_REF changes SNOMIN nitrification.
    """
    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    crop, soil, site, _agro = _providers()
    baseline = ParameterProvider(cropdata=crop, soildata=soil, sitedata=site)
    baseline.set_active_crop(_CROP_NAME, _VARIETY_NAME, "sowing", "harvest")
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
    scalar_outputs = []
    for index in range(2):
        overrides = {
            name: torch.tensor(pair[index], dtype=torch.float64)
            for name, pair in member_values.items()
        }
        scalar_outputs.append(_run_diff(overrides))

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


def _batch_member(output, index):
    """Slice one member out of a batched engine output series."""
    member = []
    for day in output:
        sliced = dict(day)
        for name in (*_DAILY, *_LAYERED):
            if name in sliced:
                sliced[name] = _member(sliced[name], index)
        member.append(sliced)
    return member


def test_mixed_ifunrn_matches_independent_runs():
    """IFUNRN 0 and 1 in one batch match separate scalar runs.

    The SNOMIN site provider defaults NOTINF to 0, which makes the fixed and
    storm-size formulas identical. A nonzero NOTINF makes IFUNRN select
    different infiltration rates on days when the PCSE rainfall is not zero.

    PCSE's layered water balance reads NINFTB when IFUNRN is 1, but never
    builds that curve. Only the IFUNRN = 0 member is compared with PCSE. The
    port uses the infiltration table from PCSE's classic water balance.
    """
    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    notinf = 0.25
    output_vars = [*_DAILY, *_LAYERED]
    pcse_fixed = _run_pcse(output_vars, site_overrides={"IFUNRN": 0, "NOTINF": notinf})
    scalar_runs = [
        _run_diff(
            {
                "IFUNRN": torch.tensor(ifunrn, dtype=torch.float64),
                "NOTINF": torch.tensor(notinf, dtype=torch.float64),
            }
        )
        for ifunrn in (0.0, 1.0)
    ]
    batch = _run_diff(
        {
            "IFUNRN": torch.tensor([0.0, 1.0], dtype=torch.float64),
            "NOTINF": torch.tensor([notinf, notinf], dtype=torch.float64),
        }
    )

    _assert_series(pcse_fixed, scalar_runs[0])
    for index, scalar in enumerate(scalar_runs):
        _assert_series(scalar, _batch_member(batch, index))

    diverged = False
    for day in batch:
        first = _numbers(_member(day["SM"], 0))
        second = _numbers(_member(day["SM"], 1))
        pairs = zip(first, second, strict=True)
        if any(not _close(left, right, tol=1e-6) for left, right in pairs):
            diverged = True
            break
    assert diverged
