import datetime
import torch
from diffwofost.ml_models.crop.partitioning import DVS_Partitioning_NN
from diffwofost.ml_models.crop.partitioning import PartitioningMLP
from diffwofost.ml_models.crop.partitioning import PartitioningNN
from diffwofost.physical_models.config import ComputeConfig
from diffwofost.physical_models.crop.partitioning import PartioningFactors
from diffwofost.physical_models.variablekiosk import VariableKiosk


def _assert_valid_partitioning(pf, expected_shape):
    assert isinstance(pf, PartioningFactors)
    assert pf.FR.shape == expected_shape
    assert pf.FL.shape == expected_shape
    assert pf.FS.shape == expected_shape
    assert pf.FO.shape == expected_shape
    assert torch.all((pf.FR >= 0.0) & (pf.FR <= 1.0))
    assert torch.all((pf.FL >= 0.0) & (pf.FL <= 1.0))
    assert torch.all((pf.FS >= 0.0) & (pf.FS <= 1.0))
    assert torch.all((pf.FO >= 0.0) & (pf.FO <= 1.0))
    assert torch.allclose(
        pf.FL + pf.FS + pf.FO,
        torch.ones(
            expected_shape, dtype=ComputeConfig.get_dtype(), device=ComputeConfig.get_device()
        ),
        atol=1e-6,
    )


class TestPartitioningModels:
    def test_partitioning_mlp_returns_valid_factors_for_scalar_input(self):
        model = PartitioningMLP()

        pf = model(torch.tensor(0.75, dtype=ComputeConfig.get_dtype()))

        _assert_valid_partitioning(pf, torch.Size([]))

    def test_partitioning_nn_returns_valid_factors_for_vector_input(self):
        model = PartitioningNN(hidden_size=16)

        pf = model(torch.tensor([0.0, 0.5, 1.0], dtype=ComputeConfig.get_dtype()))

        _assert_valid_partitioning(pf, torch.Size([3]))

    def test_partitioning_models_produce_different_outputs(self):
        dvs = torch.tensor([0.25, 1.25], dtype=ComputeConfig.get_dtype())

        mlp_pf = PartitioningMLP()(dvs)
        nn_pf = PartitioningNN(hidden_size=16)(dvs)

        assert not torch.allclose(mlp_pf.FR, nn_pf.FR)


class TestDVSPartitioningNN:
    def test_wrapper_initializes_and_updates_states_from_kiosk_dvs(self):
        class ToyPartitionModel(torch.nn.Module):
            def forward(self, dvs):
                dvs = torch.as_tensor(
                    dvs,
                    dtype=ComputeConfig.get_dtype(),
                    device=ComputeConfig.get_device(),
                )
                fr = torch.sigmoid(dvs - 1.0)
                shoots = torch.softmax(
                    torch.stack((1.0 - dvs, dvs, 0.5 * dvs), dim=-1),
                    dim=-1,
                )
                return PartioningFactors(
                    FR=fr,
                    FL=shoots[..., 0],
                    FS=shoots[..., 1],
                    FO=shoots[..., 2],
                )

        day_1 = datetime.date(2000, 1, 1)
        day_2 = datetime.date(2000, 1, 2)
        kiosk = VariableKiosk(
            [
                {"DAY": day_1, "DVS": torch.tensor(0.1, dtype=ComputeConfig.get_dtype())},
                {"DAY": day_2, "DVS": torch.tensor(1.4, dtype=ComputeConfig.get_dtype())},
            ]
        )
        kiosk(day_1)

        component = DVS_Partitioning_NN(day_1, kiosk, ToyPartitionModel())
        initial_pf = component.calc_rates(day_1, drv=None)

        kiosk(day_2)
        component.integrate(day_2)
        updated_pf = component.calc_rates(day_2, drv=None)

        _assert_valid_partitioning(initial_pf, torch.Size([]))
        _assert_valid_partitioning(updated_pf, torch.Size([]))
        assert isinstance(component.states.PF, PartioningFactors)
        assert not torch.allclose(initial_pf.FR, updated_pf.FR)
