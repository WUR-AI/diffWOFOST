import pytest
import torch
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.phenology import DVS_Phenology
from diffwofost.physical_models.engine import Engine
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from . import phy_data_folder

config = Configuration(
    CROP=DVS_Phenology,
    OUTPUT_VARS=["DVR", "DVS", "TSUM", "TSUME", "VERN"],
)


def _get_engine_inputs():
    test_data_url = f"{phy_data_folder}/test_phenology_wofost72_05.yaml"
    test_data = get_test_data(test_data_url)
    crop_model_params = [
        "TSUMEM",
        "TBASEM",
        "TEFFMX",
        "TSUM1",
        "TSUM2",
        "IDSL",
        "DLO",
        "DLC",
        "DVSI",
        "DVSEND",
        "DTSMTB",
        "VERNSAT",
        "VERNBASE",
        "VERNDVS",
    ]
    return test_data, prepare_engine_input(test_data, crop_model_params)


@pytest.mark.usefixtures("fast_mode")
class TestEngine:
    def test_engine(self):
        _, (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            _get_engine_inputs()
        )
        engine = Engine(
            parameterprovider=crop_model_params_provider,
            weatherdataprovider=weather_data_provider,
            agromanagement=agro_management_inputs,
            config=config,
        )
        start_day = engine.day
        engine.run(days=5)

        assert engine.day > start_day

    def test_engine_setup_reuses_instance(self):
        _, (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            _get_engine_inputs()
        )
        engine = Engine(config=config)

        engine.setup(crop_model_params_provider, weather_data_provider, agro_management_inputs)
        start_day = engine.day
        engine.run(days=5)
        assert engine.day > start_day

        updated_dvsi = crop_model_params_provider["DVSI"] + torch.tensor(
            0.2,
            dtype=crop_model_params_provider["DVSI"].dtype,
            device=crop_model_params_provider["DVSI"].device,
        )
        crop_model_params_provider.set_override("DVSI", updated_dvsi, check=False)

        returned_engine = engine.setup(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
        )

        assert returned_engine is engine
        assert engine.flag_terminate is False
        assert engine.day == start_day
        assert torch.equal(engine.parameterprovider["DVSI"], updated_dvsi)

        engine.run(days=1)
        assert engine.day > start_day

    def test_engine_preserves_parameter_overrides_after_run(self):
        _, (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            _get_engine_inputs()
        )
        original_dvsi = crop_model_params_provider["DVSI"].clone()

        engine = Engine(
            parameterprovider=crop_model_params_provider,
            weatherdataprovider=weather_data_provider,
            agromanagement=agro_management_inputs,
            config=config,
        )
        engine.run(days=5)

        assert torch.equal(crop_model_params_provider["DVSI"], original_dvsi)
