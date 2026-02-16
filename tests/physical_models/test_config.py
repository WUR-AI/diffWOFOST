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
    def test_default_device(self):
        ComputeConfig.reset_to_defaults()
        device = ComputeConfig.get_device()
        assert device == torch.get_default_device()

    def test_default_dtype(self):
        ComputeConfig.reset_to_defaults()
        dtype = ComputeConfig.get_dtype()
        assert dtype == torch.get_default_dtype()

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

        assert device == torch.get_default_device()
        assert dtype == torch.get_default_dtype()

    def test_models_capture_config_at_initialization(self):
        """Test that models capture the device/dtype at initialization time."""
        import datetime
        from pcse.base.variablekiosk import VariableKiosk
        from diffwofost.physical_models.crop.leaf_dynamics import WOFOST_Leaf_Dynamics

        # Setup mocks
        day = datetime.date(2000, 1, 1)

        class MockKiosk(VariableKiosk):
            pass

        mock_kiosk = MockKiosk()
        mock_kiosk.update(
            {
                "FL": 0.5,
                "FR": 0.5,
                "DVS": 0.5,
                "SAI": 0.5,
                "PAI": 0.5,
                "ADMI": 0.5,
                "RFTRA": 1.0,
                "RF_FROST": 1.0,
            }
        )

        mock_parvalues = {
            "RGRLAI": torch.tensor(0.01),
            "SPAN": torch.tensor(30.0),
            "TBASE": torch.tensor(5.0),
            "PERDL": torch.tensor(0.05),
            "TDWI": torch.tensor(50.0),
            "SLATB": [0.0, 20.0, 2.0, 20.0],
            "KDIFTB": [0.0, 0.6, 2.0, 0.6],
        }

        # 1. Config = float32
        ComputeConfig.set_dtype(torch.float32)
        model1 = WOFOST_Leaf_Dynamics(day, mock_kiosk, mock_parvalues)

        # 2. Config = float64
        ComputeConfig.set_dtype(torch.float64)
        mock_kiosk2 = MockKiosk()
        mock_kiosk2.update(
            {
                "FL": 0.5,
                "FR": 0.5,
                "DVS": 0.5,
                "SAI": 0.5,
                "PAI": 0.5,
                "ADMI": 0.5,
                "RFTRA": 1.0,
                "RF_FROST": 1.0,
            }
        )
        model2 = WOFOST_Leaf_Dynamics(day, mock_kiosk2, mock_parvalues)

        # 3. Assertions
        assert model1.dtype == torch.float32, "Model 1 should retain float32"
        assert model2.dtype == torch.float64, "Model 2 should use float64"
        assert model1.states.LV[0].dtype == torch.float32, "Model 1 states should be float32"
        assert model2.states.LV[0].dtype == torch.float64, "Model 2 states should be float64"

        # Cleanup
        ComputeConfig.reset_to_defaults()
