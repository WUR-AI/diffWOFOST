"""WOFOST 8.1 potential production matches PCSE each day.

The crop is the 8.1 wheat from PCSE's parameter repository. Soil moisture
comes from PCSE's ``DummySoilDataProvider``, which is the soil used when
production does not depend on a real profile. Site CO2 is the documented
default of the WOFOST 8.1 classic site provider. Weather is the first month
of a PCSE 7.2 potential-production series.
"""

import datetime as dt
import tempfile
from pathlib import Path
import torch
from pcse.base import ParameterProvider as PcseParameterProvider
from pcse.base import WeatherDataContainer
from pcse.base import WeatherDataProvider
from pcse.input import DummySoilDataProvider
from pcse.input import YAMLAgroManagementReader
from pcse.input import YAMLCropDataProvider
from pcse.models import Wofost81_PP
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.wofost81 import Wofost81
from diffwofost.physical_models.engine import Engine
from diffwofost.physical_models.parameter_providers import ParameterProvider
from diffwofost.physical_models.soil.n_soil_dynamics import N_PotentialProduction
from diffwofost.physical_models.soil.soil_wrappers import SoilModuleWrapper_PP
from diffwofost.physical_models.test import calculate_numerical_grad
from diffwofost.physical_models.test import get_test_data
from diffwofost.physical_models.variablekiosk import VariableKiosk

_WEATHER = (
    Path(__file__).resolve().parents[1] / "test_data" / "test_potentialproduction_wofost72_05.yaml"
)
_N_DAYS = 31
_CROP_NAME = "wheat"
_VARIETY_NAME = "Winter_wheat_101"
# Documented default of WOFOST81SiteDataProvider_Classic.
_CO2 = 360.0
_DAILY = (
    "DVS",
    "LAI",
    "TAGP",
    "TWLV",
    "TWST",
    "TWSO",
    "TRA",
    "SM",
    "NAVAIL",
    "NuptakeTotal",
    "NamountLV",
    "NamountST",
    "NamountRT",
    "NamountSO",
)


def _weather_rows():
    """First month of weather from a PCSE potential-production test file."""
    rows = []
    for row in get_test_data(_WEATHER)["WeatherVariables"][:_N_DAYS]:
        item = {key: value for key, value in row.items() if key != "SNOWDEPTH"}
        item["DTEMP"] = 0.5 * (item["TEMP"] + item["TMAX"])
        rows.append(item)
    return rows


class _CallableWeather(WeatherDataProvider):
    def __init__(self, rows):
        super().__init__()
        for row in rows:
            container = WeatherDataContainer(**row)
            self._store_WeatherDataContainer(container, container.DAY)


def _agro():
    start = _weather_rows()[0]["DAY"]
    harvest = start + dt.timedelta(days=300)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        agro_path = Path(handle.name)
        handle.write(
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
    try:
        return YAMLAgroManagementReader(str(agro_path))
    finally:
        agro_path.unlink(missing_ok=True)


def _inputs():
    crop = YAMLCropDataProvider(Wofost81_PP)
    return crop, DummySoilDataProvider(), {"CO2": _CO2}, _agro()


def _numbers(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().reshape(-1)]
    if hasattr(value, "tolist"):
        raw = value.tolist()
        return [float(raw)] if not isinstance(raw, list) else [float(item) for item in raw]
    return [float(value)]


def _close(actual, expected):
    return abs(actual - expected) <= 1e-4 + 1e-4 * abs(expected)


def _run_pcse():
    crop, soil, site, agro = _inputs()
    model = Wofost81_PP(
        PcseParameterProvider(cropdata=crop, soildata=soil, sitedata=site),
        _CallableWeather(_weather_rows()),
        agro,
        output_vars=list(_DAILY),
    )
    model.run_till(_weather_rows()[-1]["DAY"])
    return model.get_output()


def _tensor_weather(rows):
    """Match the WOFOST 7.2 engine tests: weather drivers are tensors, dates stay dates."""
    dtype = ComputeConfig.get_dtype()
    device = ComputeConfig.get_device()
    converted = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if key == "DAY":
                item[key] = value
            else:
                item[key] = torch.tensor(value, dtype=dtype, device=device)
        converted.append(item)
    return converted


def _run_diff(overrides=None):
    crop, soil, site, agro = _inputs()
    params = ParameterProvider(cropdata=crop, soildata=soil, sitedata=site)
    params.set_active_crop(_CROP_NAME, _VARIETY_NAME, "sowing", "harvest")
    for name, value in (overrides or {}).items():
        params[name] = value
    config = Configuration(
        CROP=Wofost81,
        SOIL=SoilModuleWrapper_PP,
        OUTPUT_VARS=list(_DAILY),
    )
    model = Engine(config)
    model.setup(params, iter(_tensor_weather(_weather_rows())), agro)
    model.run_till(_weather_rows()[-1]["DAY"])
    return model.get_output()


def test_daily_outputs_match_pcse():
    """Every simulated day matches PCSE, including the fixed nitrogen pool."""
    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    reference = _run_pcse()
    model = _run_diff()
    assert len(reference) == len(model)
    for ref_day, diff_day in zip(reference, model, strict=True):
        assert ref_day["day"] == diff_day["day"]
        for name in _DAILY:
            ref_values = _numbers(ref_day[name])
            diff_values = _numbers(diff_day[name])
            assert len(ref_values) == len(diff_values), (name, ref_day["day"])
            for actual, expected in zip(diff_values, ref_values, strict=True):
                assert _close(actual, expected), (name, ref_day["day"], actual, expected)
        assert _numbers(ref_day["NAVAIL"]) == [100.0]
        assert _numbers(diff_day["NAVAIL"]) == [100.0]


def test_navail_is_republished_after_the_kiosk_flush():
    """The engine drops published states during integrate. touch() puts 100 back."""
    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    kiosk = VariableKiosk()
    module = N_PotentialProduction(dt.date(2010, 1, 1), kiosk, {})
    assert float(kiosk["NAVAIL"]) == 100.0

    kiosk.flush_states()
    assert "NAVAIL" not in kiosk

    module.integrate(dt.date(2010, 1, 2))
    assert float(module.states.NAVAIL) == 100.0
    assert float(kiosk["NAVAIL"]) == 100.0


class _LastDay:
    """Run one month and return the final-day outputs."""

    def __call__(self, overrides):
        last = _run_diff(overrides)[-1]
        result = {}
        for name, value in last.items():
            if isinstance(value, torch.Tensor):
                result[name] = value
            elif value is not None and name == "TAGP":
                result[name] = torch.tensor(value, dtype=torch.float64)
        return result


def test_autograd_matches_numerical_gradient():
    """Above-ground biomass keeps a gradient with respect to leaf conversion."""
    ComputeConfig.set_dtype(torch.float64)
    ComputeConfig.set_device("cpu")
    crop, soil, site, _agro_reader = _inputs()
    baseline = ParameterProvider(cropdata=crop, soildata=soil, sitedata=site)
    baseline.set_active_crop(_CROP_NAME, _VARIETY_NAME, "sowing", "harvest")
    param = torch.nn.Parameter(torch.tensor(float(baseline["CVL"]), dtype=torch.float64))

    numerical = calculate_numerical_grad(lambda: _LastDay(), "CVL", param, "TAGP")
    loss = _LastDay()({"CVL": param})["TAGP"].sum()
    autograd = torch.autograd.grad(loss, param)[0]
    assert torch.isfinite(autograd).all()
    assert torch.any(autograd != 0)
    torch.testing.assert_close(numerical, autograd, rtol=1e-3, atol=1e-3)
