"""WOFOST 8.1 + SNOMIN matches PCSE and is differentiable."""

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

_GYM = Path("/home/michiel/WUR/PCSE-Gym/pcse_gym/envs/configs")
_START = dt.date(2000, 3, 15)
_END = dt.date(2000, 4, 14)


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


def _scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0])
    if hasattr(value, "shape"):
        return float(value.reshape(-1)[0])
    return float(value)


def _close(actual, expected, tol=1e-4):
    return abs(actual - expected) <= tol + tol * abs(expected)


@pytest.mark.skipif(not _GYM.exists(), reason="PCSE-Gym crop and soil files are not available")
def test_wofost81_snomin_matches_pcse_and_has_gradients():
    """A month of WOFOST 8.1 SNOMIN stays close to PCSE and yields a gradient."""
    ComputeConfig.set_dtype(torch.float64)
    crop, soil, site, agro = _providers()
    rows = _weather_rows()
    reference = Wofost81_NWLP_MLWB_SNOMIN(
        PcseParameterProvider(cropdata=crop, soildata=soil, sitedata=site),
        _CallableWeather(rows),
        agro,
    )
    reference.run_till(_END)

    config = Configuration(
        CROP=Wofost81,
        SOIL=SoilModuleWrapper_NWLP_MLWB_SNOMIN,
        OUTPUT_VARS=["DVS", "LAI", "TAGP", "SM", "NAVAIL", "TRA"],
    )
    # Rebuild providers: the reference run activates and may mutate the crop provider.
    crop, soil, site, agro = _providers()
    model = Engine(config)
    model.setup(
        ParameterProvider(cropdata=crop, soildata=soil, sitedata=site),
        iter(rows),
        agro,
    )
    model.run_till(_END)

    ref_last = reference.get_output()[-1]
    diff_last = model.get_output()[-1]
    for name in ("DVS", "LAI", "TAGP", "TRA", "NAVAIL"):
        assert _close(_scalar(diff_last[name]), _scalar(ref_last[name])), (
            name,
            _scalar(diff_last[name]),
            _scalar(ref_last[name]),
        )
    ref_sm = ref_last["SM"]
    diff_sm = diff_last["SM"]
    for layer in range(len(ref_sm)):
        assert _close(_scalar(diff_sm[layer]), _scalar(ref_sm[layer])), (
            "SM",
            layer,
            _scalar(diff_sm[layer]),
            _scalar(ref_sm[layer]),
        )

    # Gradient of above-ground biomass w.r.t. a conversion efficiency.
    crop, soil, site, agro = _providers()
    params = ParameterProvider(cropdata=crop, soildata=soil, sitedata=site)
    params.set_active_crop("winterwheat", "Arminda", "sowing", "harvest")
    cvl = torch.tensor(float(params["CVL"]), dtype=torch.float64, requires_grad=True)
    params["CVL"] = cvl
    model = Engine(config)
    model.setup(params, iter(_weather_rows()), agro)
    model.run_till(_END)
    tagp = model.get_output()[-1]["TAGP"]
    if not isinstance(tagp, torch.Tensor):
        tagp = torch.tensor(tagp, dtype=torch.float64)
    tagp.sum().backward()
    assert cvl.grad is not None
    assert torch.isfinite(cvl.grad)
