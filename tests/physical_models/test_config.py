import torch
from pcse.agromanager import AgroManager
from pcse.crop.phenology import DVS_Phenology
from pcse.soil.classic_waterbalance import WaterbalancePP
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.leaf_dynamics import WOFOST_Leaf_Dynamics
from . import phy_data_folder


class TestConfiguration:
    def test_basic_config_requires_only_crop_model(self):
        config = Configuration(CROP=WOFOST_Leaf_Dynamics)
        assert isinstance(config, Configuration)

    def test_config_accept_other_optional_input_args(self):
        config = Configuration(
            CROP=WOFOST_Leaf_Dynamics,
            SOIL=WaterbalancePP,
            AGROMANAGEMENT=AgroManager,
            OUTPUT_VARS=[],
            SUMMARY_OUTPUT_VARS=[],
            TERMINAL_OUTPUT_VARS=[],
            OUTPUT_INTERVAL="weekly",
            OUTPUT_INTERVAL_DAYS=1,
            OUTPUT_WEEKDAY=0,
            model_config_file=None,
            description="this is the description",
        )
        assert isinstance(config, Configuration)

    def test_config_can_be_instantiated_from_a_default_pcse_config_file(self):
        config = Configuration.from_pcse_config_file("Wofost72_Pheno.conf")
        assert config.SOIL is None
        assert config.CROP == DVS_Phenology
        assert config.AGROMANAGEMENT == AgroManager

    def test_config_can_be_instantiated_from_a_custom_pcse_config_file(self):
        config_file_path = phy_data_folder / "Wofost72_Pheno_test.conf"
        config = Configuration.from_pcse_config_file(config_file_path)
        assert isinstance(config, Configuration)
        assert config.model_config_file == config_file_path.resolve()
        assert config.description is not None  # Description is parsed from the module docstring

    def test_output_variables_can_be_updated(self):
        config = Configuration(CROP=WOFOST_Leaf_Dynamics)
        assert not config.OUTPUT_VARS
        assert not config.SUMMARY_OUTPUT_VARS
        assert not config.TERMINAL_OUTPUT_VARS
        # Test all accepted data types
        config.update_output_variable_lists(
            output_vars=["DVS", "LAI"],  # list
            summary_vars="LAI",  # str
            terminal_vars={"DVS"},  # set
        )
        assert config.OUTPUT_VARS == ["DVS", "LAI"]
        assert config.SUMMARY_OUTPUT_VARS == ["LAI"]
        assert config.TERMINAL_OUTPUT_VARS == ["DVS"]


class TestComputeConfig:
    def test_default_device_is_cuda_or_cpu(self):
        ComputeConfig.reset_to_defaults()
        device = ComputeConfig.get_device()
        assert device.type in ["cpu", "cuda"]

    def test_default_dtype_is_float64(self):
        ComputeConfig.reset_to_defaults()
        dtype = ComputeConfig.get_dtype()
        assert dtype == torch.float64

    def test_set_device_with_string(self):
        ComputeConfig.set_device("cpu")
        device = ComputeConfig.get_device()
        assert device.type == "cpu"

    def test_set_device_with_torch_device(self):
        ComputeConfig.set_device(torch.device("cpu"))
        device = ComputeConfig.get_device()
        assert device.type == "cpu"

    def test_set_dtype(self):
        ComputeConfig.set_dtype(torch.float32)
        dtype = ComputeConfig.get_dtype()
        assert dtype == torch.float32

    def test_reset_to_defaults(self):
        ComputeConfig.set_device("cpu")
        ComputeConfig.set_dtype(torch.float32)
        ComputeConfig.reset_to_defaults()

        device = ComputeConfig.get_device()
        dtype = ComputeConfig.get_dtype()

        assert device.type in ["cpu", "cuda"]
        assert dtype == torch.float64
