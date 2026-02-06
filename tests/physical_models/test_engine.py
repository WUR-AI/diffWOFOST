from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.phenology import DVS_Phenology
from diffwofost.physical_models.engine import Engine
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from . import phy_data_folder

config = Configuration(CROP=DVS_Phenology)


class TestEngine:
    def test_engine_can_be_instantiated_from_default_pcse_config(self):
        test_data_url = f"{phy_data_folder}/test_phenology_wofost72_01.yaml"
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
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data, crop_model_params)
        )
        engine = Engine(
            parameterprovider=crop_model_params_provider,
            weatherdataprovider=weather_data_provider,
            agromanagement=agro_management_inputs,
            config=config,
        )
        assert isinstance(engine, Engine)
