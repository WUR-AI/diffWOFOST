from datetime import date
from pathlib import Path
from unittest.mock import Mock
import pytest
import torch
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.phenology import DVS_Phenology
from diffwofost.physical_models.crop.wofost72 import Wofost72
from diffwofost.physical_models.engine import Engine
from diffwofost.physical_models.engine import _get_params_shape
from diffwofost.physical_models.soil.classic_waterbalance import WaterbalancePP
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


class _DummyParameterProvider(dict):
    def __init__(self, values):
        super().__init__(values)
        self._unique_parameters = list(values)

    def set_active_crop(self, *args):
        self.active_crop_args = args


@pytest.mark.usefixtures("fast_mode")
class TestEngine:
    def test_engine_requires_config(self):
        with pytest.raises(TypeError, match="A model configuration must be provided"):
            Engine()

    def test_engine_loads_configuration_from_path(self, monkeypatch):
        config_path = Path("dummy_config.py")
        loaded_paths = []

        def fake_loader(path):
            loaded_paths.append(path)
            return config

        monkeypatch.setattr(
            Configuration,
            "from_pcse_config_file",
            staticmethod(fake_loader),
        )

        engine = Engine(config=config_path)

        assert engine.mconf is config
        assert loaded_paths == [config_path]

    def test_engine(self):
        _, (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            _get_engine_inputs()
        )
        engine = Engine(config=config)
        engine.setup(crop_model_params_provider, weather_data_provider, agro_management_inputs)
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

        engine = Engine(config=config)
        engine.setup(crop_model_params_provider, weather_data_provider, agro_management_inputs)
        engine.run(days=5)

        assert torch.equal(crop_model_params_provider["DVSI"], original_dvsi)

    def test_setup_initializes_soil_component_when_configured(self):
        soil_config = Configuration(
            CROP=Wofost72,
            SOIL=WaterbalancePP,
            OUTPUT_VARS=["SM", "EVS"],
        )
        test_data_url = f"{phy_data_folder}/test_potentialproduction_wofost72_05.yaml"
        test_data = get_test_data(test_data_url)
        parameterprovider, weatherdataprovider, agromanagement, _ = prepare_engine_input(
            test_data,
            ["SMFCF"],
        )
        engine = Engine(config=soil_config)

        returned_engine = engine.setup(parameterprovider, weatherdataprovider, agromanagement)

        assert returned_engine is engine
        assert engine.soil is not None

    def test_on_crop_start_raises_when_crop_is_already_active(self):
        _, (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            _get_engine_inputs()
        )
        engine = Engine(config=config)
        engine.setup(crop_model_params_provider, weather_data_provider, agro_management_inputs)

        with pytest.raises(RuntimeError, match="A CROP_START signal was received"):
            engine._on_CROP_START(engine.day)

    def test_finish_cropsimulation_deletes_crop_when_requested(self):
        _, (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            _get_engine_inputs()
        )
        engine = Engine(config=config)
        engine.setup(crop_model_params_provider, weather_data_provider, agro_management_inputs)
        crop = engine.crop
        crop.finalize = Mock()
        crop._delete = Mock()
        engine.flag_crop_finish = True
        engine.flag_crop_delete = True
        engine._save_summary_output = Mock()

        engine._finish_cropsimulation(date(2000, 1, 1))

        assert engine.flag_crop_finish is False
        assert engine.flag_crop_delete is False
        crop.finalize.assert_called_once_with(date(2000, 1, 1))
        engine._save_summary_output.assert_called_once_with()
        crop._delete.assert_called_once_with()
        assert engine.crop is None

    def test_finish_cropsimulation_keeps_crop_when_not_deleting(self):
        _, (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            _get_engine_inputs()
        )
        engine = Engine(config=config)
        engine.setup(crop_model_params_provider, weather_data_provider, agro_management_inputs)
        crop = engine.crop
        crop.finalize = Mock()
        crop._delete = Mock()
        engine.flag_crop_finish = True
        engine.flag_crop_delete = False
        engine._save_summary_output = Mock()

        engine._finish_cropsimulation(date(2000, 1, 1))

        assert engine.flag_crop_finish is False
        assert engine.flag_crop_delete is False
        crop.finalize.assert_called_once_with(date(2000, 1, 1))
        engine._save_summary_output.assert_called_once_with()
        crop._delete.assert_not_called()
        assert engine.crop is crop

    def test_get_params_shape_uses_first_tensor_shape(self):
        parameterprovider = _DummyParameterProvider(
            {
                "A": torch.tensor([1.0, 2.0]),
                "B": 3.0,
            }
        )

        assert _get_params_shape(parameterprovider) == (2,)

    def test_get_params_shape_raises_on_mismatched_shapes(self):
        parameterprovider = _DummyParameterProvider(
            {
                "A": torch.ones(2),
                "B": torch.ones(3),
            }
        )

        with pytest.raises(ValueError, match="Non-matching shapes found in parameter provider"):
            _get_params_shape(parameterprovider)

    def test_get_params_shape_returns_empty_tuple_for_scalar_parameters(self):
        parameterprovider = _DummyParameterProvider(
            {
                "A": 1.0,
                "B": 2.0,
            }
        )

        assert _get_params_shape(parameterprovider) == ()
