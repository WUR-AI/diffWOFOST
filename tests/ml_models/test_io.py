import tempfile
from pathlib import Path
import torch
from diffwofost.ml_models import load_model
from diffwofost.ml_models import save_model
from diffwofost.ml_models.crop.partitioning import PartitioningMLP
from diffwofost.ml_models.crop.partitioning import PartitioningNN
from diffwofost.physical_models.config import ComputeConfig


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


def test_partitioning_nn_round_trips_through_safetensors(tmp_path):
    model = PartitioningNN(hidden_size=16)
    _fill_parameters(model, scale=10.0)

    path = save_model(model, tmp_path / "partitioning_nn.safetensors")
    restored = load_model(path, model_class=PartitioningNN)
    dvs = torch.tensor([0.0, 0.5, 1.0], dtype=ComputeConfig.get_dtype())

    assert restored.hidden_size == 16
    _assert_same_partition_outputs(model, restored, dvs)


def test_partitioning_mlp_uses_stable_default_save_path():
    model = PartitioningMLP(hidden_size=12)
    _fill_parameters(model, scale=20.0)

    first_path = save_model(model)
    second_path = save_model(model)
    restored = load_model(first_path)
    dvs = torch.tensor([0.2, 1.4], dtype=ComputeConfig.get_dtype())

    assert isinstance(restored, PartitioningMLP)
    assert restored.hidden_size == 12
    assert first_path == second_path
    assert first_path.parent == Path(tempfile.gettempdir()) / "diffwofost-ml-models"
    assert first_path.suffix == ".safetensors"
    _assert_same_partition_outputs(model, restored, dvs)


def test_partitioning_mlp_can_override_default_save_name(tmp_path):
    model = PartitioningMLP(hidden_size=12)
    _fill_parameters(model, scale=20.0)

    path = save_model(
        model,
        directory=tmp_path,
        filename="partitioning_mlp_custom.safetensors",
    )
    restored = load_model(path)
    dvs = torch.tensor([0.2, 1.4], dtype=ComputeConfig.get_dtype())

    assert isinstance(restored, PartitioningMLP)
    assert restored.hidden_size == 12
    assert path == tmp_path / "partitioning_mlp_custom.safetensors"
    _assert_same_partition_outputs(model, restored, dvs)
