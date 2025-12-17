from pcse.agromanager import AgroManager
from pcse.crop.phenology import DVS_Phenology
from pcse.soil.classic_waterbalance import WaterbalancePP
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
