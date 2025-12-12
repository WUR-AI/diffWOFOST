from pcse.agromanager import AgroManager
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

    def test_config_can_be_instantiated_from_a_pcse_config_file(self):
        config_file_path = phy_data_folder / "WOFOST_Leaf_Dynamics.conf"
        config = Configuration.from_pcse_config_file(config_file_path)
        assert isinstance(config, Configuration)
        assert config.model_config_file == config_file_path.resolve()

    def test_output_variables_can_be_updated(self):
        config = Configuration(CROP=WOFOST_Leaf_Dynamics)
        assert not config.OUTPUT_VARS
        config.update_output_variable_lists(output_vars=["DVS", "LAI"])
        assert config.OUTPUT_VARS == ["DVS", "LAI"]
