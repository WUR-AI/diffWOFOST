from pathlib import Path
import torch
from pcse.base.parameter_providers import ParameterProvider
from diffwofost import load_model
from diffwofost import save_model
from diffwofost.ml_models.crop.partitioning import DVS_Partitioning_NN
from diffwofost.ml_models.crop.partitioning import PartitioningMLP
from diffwofost.ml_models.crop.partitioning import PartitioningNN
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.partitioning import DVS_Partitioning
from diffwofost.physical_models.crop.wofost72 import Wofost72
from diffwofost.physical_models.soil.classic_waterbalance import WaterbalancePP
from diffwofost.physical_models.test import EngineTestHelper
from diffwofost.physical_models.test import get_test_data
from diffwofost.physical_models.test import prepare_engine_input
from .physical_models import phy_data_folder


def _fill_parameters(model, scale):
    for index, parameter in enumerate(model.parameters(), start=1):
        parameter.data.fill_(index / scale)


def _assert_same_partition_outputs(left, right, dvs):
    left_pf = left(dvs)
    right_pf = right(dvs)

    assert torch.allclose(left_pf.FR, right_pf.FR)
    assert torch.allclose(left_pf.FL, right_pf.FL)
    assert torch.allclose(left_pf.FS, right_pf.FS)
    assert torch.allclose(left_pf.FO, right_pf.FO)


def test_save_and_load_partitioning_nn(tmp_path):
    model = PartitioningNN(hidden_size=16)
    _fill_parameters(model, scale=10.0)

    path = save_model(tmp_path / "partitioning_nn.safetensors", model=model)
    restored = load_model(path, model_class=PartitioningNN)
    dvs = torch.tensor([0.0, 0.5, 1.0], dtype=ComputeConfig.get_dtype())

    assert restored.hidden_size == 16
    _assert_same_partition_outputs(model, restored, dvs)


def test_partitioning_mlp_uses_stable_default_save_path():
    model = PartitioningMLP(hidden_size=12)
    _fill_parameters(model, scale=20.0)

    first_path = save_model(model=model)
    second_path = save_model(model=model)
    restored = load_model(first_path)
    dvs = torch.tensor([0.2, 1.4], dtype=ComputeConfig.get_dtype())

    assert isinstance(restored, PartitioningMLP)
    assert restored.hidden_size == 12
    assert first_path == second_path
    assert first_path.parent == Path(__file__).resolve().parents[1] / ".diffwofost-ml-models"
    assert first_path.suffix == ".safetensors"
    _assert_same_partition_outputs(model, restored, dvs)


def test_partitioning_mlp_can_override_default_save_name(tmp_path):
    model = PartitioningMLP(hidden_size=12)
    _fill_parameters(model, scale=20.0)

    path = save_model(
        directory=tmp_path,
        filename="partitioning_mlp_custom.safetensors",
        model=model,
    )
    restored = load_model(path)
    dvs = torch.tensor([0.2, 1.4], dtype=ComputeConfig.get_dtype())

    assert isinstance(restored, PartitioningMLP)
    assert restored.hidden_size == 12
    assert path == tmp_path / "partitioning_mlp_custom.safetensors"
    _assert_same_partition_outputs(model, restored, dvs)


# ---------------------------------------------------------------------------
# Physical model (Configuration + ParameterProvider) tests
# ---------------------------------------------------------------------------


def _make_simple_config():
    """Return a minimal ``Configuration`` and ``ParameterProvider`` for testing."""
    config = Configuration(
        CROP=DVS_Partitioning,
        OUTPUT_VARS=["FR", "FL", "FS", "FO"],
    )
    provider = ParameterProvider(
        cropdata={
            "FRTB": [0.0, 0.5, 0.0, 0.1],
            "FLTB": [0.0, 0.5, 0.5, 0.1],
            "FSTB": [0.0, 0.5, 0.5, 0.1],
            "FOTB": [0.0, 0.5, 0.0, 0.1],
        },
    )
    # Set a tensor override to exercise the safetensors path
    provider.set_override(
        "FRTB",
        torch.tensor([0.0, 0.5, 0.0, 0.1], dtype=ComputeConfig.get_dtype()),
        check=False,
    )
    return config, provider


def test_save_and_load_preserves_config_metadata(tmp_path):
    """Configuration class refs and output vars should survive save→load."""
    config, provider = _make_simple_config()

    saved = save_model(tmp_path / "model", config=config, parameterprovider=provider)
    loaded_config, _ = load_model(saved)

    assert loaded_config.CROP is DVS_Partitioning
    assert loaded_config.OUTPUT_VARS == ["FR", "FL", "FS", "FO"]
    assert loaded_config.SOIL is None
    assert loaded_config.CROP_NN_MODEL is None


def test_save_and_load_preserves_parameter_values(tmp_path):
    """Scalar and tensor parameter values should survive save→load."""
    config, provider = _make_simple_config()
    dtype = ComputeConfig.get_dtype()

    saved = save_model(tmp_path / "model", config=config, parameterprovider=provider)
    _, loaded_provider = load_model(saved)

    # The override was a tensor
    assert "FRTB" in loaded_provider
    assert torch.allclose(
        loaded_provider["FRTB"],
        torch.tensor([0.0, 0.5, 0.0, 0.1], dtype=dtype),
    )
    # FLTB should still be the original list (non-tensor)
    assert loaded_provider["FLTB"] == [0.0, 0.5, 0.5, 0.1]


def test_save_and_load_with_soil_config(tmp_path):
    """Configuration with a SOIL class should survive save→load."""
    config = Configuration(
        CROP=Wofost72,
        SOIL=WaterbalancePP,
        OUTPUT_VARS=["LAI", "TWSO"],
    )
    provider = ParameterProvider(
        cropdata={"TDWI": 0.5},
        soildata={"SMW": 0.3},
    )

    saved = save_model(tmp_path / "model", config=config, parameterprovider=provider)
    loaded_config, loaded_provider = load_model(saved)

    assert loaded_config.CROP is Wofost72
    assert loaded_config.SOIL is WaterbalancePP
    assert loaded_config.OUTPUT_VARS == ["LAI", "TWSO"]
    assert loaded_provider["TDWI"] == 0.5
    assert loaded_provider["SMW"] == 0.3


def test_save_and_load_with_crop_nn_model(tmp_path):
    """CROP_NN_MODEL (an nn.Module instance) should survive save→load."""
    model = PartitioningNN(hidden_size=8)
    _fill_parameters(model, scale=10.0)

    config = Configuration(
        CROP=DVS_Partitioning_NN,
        CROP_NN_MODEL=model,
        OUTPUT_VARS=["FR", "FL", "FS", "FO"],
    )
    provider = ParameterProvider(cropdata={"TDWI": 0.5})

    saved = save_model(tmp_path / "model", config=config, parameterprovider=provider)
    loaded_config, loaded_provider = load_model(saved)

    assert loaded_config.CROP is DVS_Partitioning_NN
    assert isinstance(loaded_config.CROP_NN_MODEL, PartitioningNN)
    assert loaded_config.CROP_NN_MODEL.hidden_size == 8

    dvs = torch.tensor([0.0, 0.5, 1.0], dtype=ComputeConfig.get_dtype())
    _assert_same_partition_outputs(model, loaded_config.CROP_NN_MODEL, dvs)

    assert loaded_provider["TDWI"] == 0.5


def test_save_and_load_with_crop_components(tmp_path):
    """CROP_COMPONENTS with an embedded ML model should survive save→load."""
    model = PartitioningNN(hidden_size=8)
    _fill_parameters(model, scale=10.0)

    config = Configuration(
        CROP=Wofost72,
        CROP_COMPONENTS={
            "partitioning": {
                "class": DVS_Partitioning_NN,
                "model": model,
            }
        },
        SOIL=WaterbalancePP,
        OUTPUT_VARS=["FR", "FL", "FS", "FO"],
    )
    provider = ParameterProvider(cropdata={"TDWI": 0.5})

    saved = save_model(tmp_path / "model", config=config, parameterprovider=provider)
    loaded_config, loaded_provider = load_model(saved)

    assert "partitioning" in loaded_config.CROP_COMPONENTS
    loaded_comp = loaded_config.CROP_COMPONENTS["partitioning"]
    assert loaded_comp["class"] is DVS_Partitioning_NN
    assert isinstance(loaded_comp["model"], PartitioningNN)
    assert loaded_comp["model"].hidden_size == 8

    dvs = torch.tensor([0.0, 0.5, 1.0], dtype=ComputeConfig.get_dtype())
    _assert_same_partition_outputs(model, loaded_comp["model"], dvs)


def test_save_and_load_produces_same_engine_results(tmp_path):
    """Save, reload, re-run engine — outputs must match the original run."""
    test_data_url = f"{phy_data_folder}/test_partitioning_wofost72_05.yaml"
    test_data = get_test_data(test_data_url)
    crop_model_params = ["FRTB", "FLTB", "FSTB", "FOTB"]

    # Original run
    config = Configuration(CROP=DVS_Partitioning, OUTPUT_VARS=["FR", "FL", "FS", "FO"])
    (
        orig_provider,
        weather_data_provider,
        agro_management_inputs,
        external_states,
    ) = prepare_engine_input(test_data, crop_model_params)

    engine = EngineTestHelper(config=config)
    engine.setup(orig_provider, weather_data_provider, agro_management_inputs, external_states)
    engine.run_till_terminate()
    original_results = engine.get_output()

    # Save & reload
    saved = save_model(tmp_path / "model", config=config, parameterprovider=orig_provider)
    loaded_config, loaded_provider = load_model(saved)

    # Run with loaded config + provider
    engine2 = EngineTestHelper(config=loaded_config)
    engine2.setup(loaded_provider, weather_data_provider, agro_management_inputs, external_states)
    engine2.run_till_terminate()
    reloaded_results = engine2.get_output()

    for orig_day, reload_day in zip(original_results, reloaded_results, strict=True):
        for var in ["FR", "FL", "FS", "FO"]:
            assert torch.allclose(orig_day[var], reload_day[var]), (
                f"Mismatch in {var} at day {orig_day.get('day')}"
            )
