import copy
from unittest.mock import patch
import pytest
import torch
from pcse.models import Wofost72_PP
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.partitioning import WOFOST_Partitioning
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from .. import phy_data_folder

partitioning_config = Configuration(CROP=WOFOST_Partitioning, OUTPUT_VARS=["FR", "FL", "FS", "FO"])


def get_test_diff_partitioning(device: str = "cpu"):
    """Build a small wrapper module for differentiable tests."""
    test_data_url = f"{phy_data_folder}/test_potentialproduction_wofost72_01.yaml"
    test_data = get_test_data(test_data_url)
    crop_model_params = ["FRTB", "FLTB", "FSTB", "FOTB"]
    (
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        external_states,
    ) = prepare_engine_input(test_data, crop_model_params, device=device)
    return DiffPartitioning(
        copy.deepcopy(crop_model_params_provider),
        weather_data_provider,
        agro_management_inputs,
        partitioning_config,
        copy.deepcopy(external_states),
        device=device,
    )


class DiffPartitioning(torch.nn.Module):
    def __init__(
        self,
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        config,
        external_states,
        device: str = "cpu",
    ):
        super().__init__()
        self.crop_model_params_provider = crop_model_params_provider
        self.weather_data_provider = weather_data_provider
        self.agro_management_inputs = agro_management_inputs
        self.config = config
        self.external_states = external_states
        self.device = device

    def forward(self, params_dict: dict[str, torch.Tensor]):
        # pass new value of parameters to the model
        for name, value in params_dict.items():
            if isinstance(value, torch.Tensor) and value.device.type != self.device:
                value = value.to(self.device)
            self.crop_model_params_provider.set_override(name, value, check=False)

        engine = EngineTestHelper(
            self.crop_model_params_provider,
            self.weather_data_provider,
            self.agro_management_inputs,
            self.config,
            self.external_states,
            device=self.device,
        )
        engine.run_till_terminate()
        results = engine.get_output()

        return {
            var: torch.stack([item[var] for item in results]) for var in ["FR", "FL", "FS", "FO"]
        }


class TestPartitioning:
    data_urls = [f"{phy_data_folder}/test_partitioning_wofost72_{i:02d}.yaml" for i in range(1, 45)]
    wofost72_data_urls = [
        f"{phy_data_folder}/test_potentialproduction_wofost72_{i:02d}.yaml" for i in range(1, 45)
    ]

    @pytest.mark.parametrize("test_data_url", data_urls)
    def test_partitioning_with_testengine(self, test_data_url, device):
        """Mirror of leaf dynamics structure: compare against YAML references."""
        test_data = get_test_data(test_data_url)
        crop_model_params = ["FRTB", "FLTB", "FSTB", "FOTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, device=device)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            partitioning_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        expected_results, expected_precision = (
            test_data["ModelResults"],
            test_data.get("Precision", {"FR": 1e-6, "FL": 1e-6, "FS": 1e-6, "FO": 1e-6}),
        )

        assert len(actual_results) == len(expected_results)
        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            # Range and checksum invariants
            for key in ("FR", "FL", "FS", "FO"):
                assert torch.all((model[key] >= 0.0) & (model[key] <= 1.0))
            checksum = model["FR"] + (model["FL"] + model["FS"] + model["FO"]) * (1.0 - model["FR"])
            assert torch.allclose(checksum, torch.ones_like(checksum), atol=1e-4)
            # Reference checks
            assert all(
                torch.all(abs(reference[var] - model[var]) < precision)
                for var, precision in expected_precision.items()
            )

    @pytest.mark.parametrize("param", ["FRTB", "FLTB", "FSTB", "FOTB"])
    def test_partitioning_with_one_parameter_vector(self, param, device):
        test_data_url = f"{phy_data_folder}/test_potentialproduction_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["FRTB", "FLTB", "FSTB", "FOTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, device=device)

        vector_tables: list[list[float]] = [
            [0.0, 0.3, 2.0, 0.1],
            [0.0, 0.2, 2.0, 0.05],
            [0.0, 0.4, 2.0, 0.2],
        ]
        crop_model_params_provider.set_override(param, vector_tables, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            partitioning_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        results = engine.get_output()

        for day in results:
            for key in ("FR", "FL", "FS", "FO"):
                assert day[key].ndim >= 1
                assert torch.all(torch.isfinite(day[key]))

    def test_partitioning_with_different_parameter_values(self, device):
        test_data_url = f"{phy_data_folder}/test_potentialproduction_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["FRTB", "FLTB", "FSTB", "FOTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, device=device)

        # Build parameter vectors with multiple values for each table
        tables_low_high = [
            [0.0, 0.2, 2.0, 0.1],
            [0.0, 0.5, 2.0, 0.3],
        ]
        for name in ("FRTB", "FLTB", "FSTB", "FOTB"):
            crop_model_params_provider.set_override(name, tables_low_high, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            partitioning_config,
            external_states,
            device=device,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        for day in actual_results:
            for key in ("FR", "FL", "FS", "FO"):
                assert day[key].shape[0] == 2
                assert torch.all(torch.isfinite(day[key]))

    def test_partitioning_with_multiple_parameter_vectors(self):
        test_data_url = f"{phy_data_folder}/test_potentialproduction_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["FRTB", "FLTB", "FSTB", "FOTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        tables = [
            [0.0, 0.3, 2.0, 0.1],
            [0.0, 0.25, 2.0, 0.15],
        ]
        for name in ("FRTB", "FLTB", "FSTB", "FOTB"):
            crop_model_params_provider.set_override(name, tables, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            partitioning_config,
            external_states,
            device="cpu",
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        for day in actual_results:
            for key in ("FR", "FL", "FS", "FO"):
                assert day[key].shape[0] == 2
                assert torch.all(torch.isfinite(day[key]))

    def test_partitioning_with_multiple_parameter_arrays(self):
        test_data_url = f"{phy_data_folder}/test_partitioning_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["FRTB", "FLTB", "FSTB", "FOTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        # Repeat AfgenTrait tables to vectorize like in leaf dynamics: shape (30, 5, K)
        for param in ("FRTB", "FLTB", "FSTB", "FOTB"):
            base = crop_model_params_provider[param]
            repeated = base.repeat(30, 5, 1)
            crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            partitioning_config,
            external_states,
            device="cpu",
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        expected_results, expected_precision = (
            test_data["ModelResults"],
            test_data.get("Precision", {"FR": 1e-6, "FL": 1e-6, "FS": 1e-6, "FO": 1e-6}),
        )

        assert len(actual_results) == len(expected_results)
        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                torch.all(abs(reference[var] - model[var]) < precision)
                for var, precision in expected_precision.items()
            )
            assert all(
                model[var].shape == (30, 5) for var in expected_precision.keys()
            )  # check the output shapes

    def test_partitioning_with_incompatible_parameter_vectors(self):
        test_data_url = f"{phy_data_folder}/test_potentialproduction_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["FRTB", "FLTB", "FSTB", "FOTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        crop_model_params_provider.set_override("FRTB", [[0.0, 0.3, 2.0, 0.1]] * 4, check=False)
        crop_model_params_provider.set_override("FLTB", [[0.0, 0.3, 2.0, 0.1]] * 2, check=False)

        with pytest.raises(AssertionError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                partitioning_config,
                external_states,
                device="cpu",
            )

    @pytest.mark.parametrize("test_data_url", wofost72_data_urls[:1])
    def test_wofost_pp_with_partitioning(self, test_data_url):
        # prepare model input
        test_data = get_test_data(test_data_url)
        crop_model_params = ["FRTB", "FLTB", "FSTB", "FOTB"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data, crop_model_params)
        )

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        with patch("pcse.crop.wofost72.Partitioning", WOFOST_Partitioning):
            model = Wofost72_PP(
                crop_model_params_provider, weather_data_provider, agro_management_inputs
            )
            model.run_till_terminate()
            actual_results = model.get_output()

            assert len(actual_results) == len(expected_results)

            for reference, model in zip(expected_results, actual_results, strict=False):
                assert reference["DAY"] == model["day"]
                assert all(
                    abs(reference[var] - model[var]) < precision
                    for var, precision in expected_precision.items()
                )


class TestDiffPartitioningGradients:
    """Gradient tests mirroring leaf dynamics test structure."""

    param_names = ["FRTB", "FLTB", "FSTB", "FOTB"]
    output_names = ["FR", "FL", "FS", "FO"]

    # parameter configurations (value, dtype)
    param_configs = {
        "single": {
            "FRTB": ([[0.0, 0.3, 2.0, 0.1]], torch.float64),
            "FLTB": ([[0.0, 0.3, 2.0, 0.1]], torch.float64),
            "FSTB": ([[0.0, 0.3, 2.0, 0.1]], torch.float64),
            "FOTB": ([[0.0, 0.3, 2.0, 0.1]], torch.float64),
        },
        "tensor": {
            "FRTB": (
                [[0.0, 0.3, 2.0, 0.1], [0.0, 0.4, 2.0, 0.2], [0.0, 0.2, 2.0, 0.05]],
                torch.float64,
            ),
            "FLTB": (
                [[0.0, 0.25, 2.0, 0.15], [0.0, 0.35, 2.0, 0.25], [0.0, 0.2, 2.0, 0.1]],
                torch.float64,
            ),
            "FSTB": (
                [[0.0, 0.5, 2.0, 0.3], [0.0, 0.4, 2.0, 0.2], [0.0, 0.6, 2.0, 0.35]],
                torch.float64,
            ),
            "FOTB": (
                [[0.0, 0.1, 2.0, 0.05], [0.0, 0.2, 2.0, 0.1], [0.0, 0.15, 2.0, 0.08]],
                torch.float64,
            ),
        },
    }

    # mapping of which outputs should have gradients for each param
    gradient_mapping = {
        "FRTB": ["FR"],
        "FLTB": ["FL"],
        "FSTB": ["FS"],
        "FOTB": ["FO"],
    }

    gradient_params: list[tuple[str, str]] = []
    no_gradient_params: list[tuple[str, str]] = []
    for p in param_names:
        for o in output_names:
            if o in gradient_mapping.get(p, []):
                gradient_params.append((p, o))
            else:
                no_gradient_params.append((p, o))

    @pytest.mark.parametrize("param_name,output_name", no_gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_no_gradients(self, param_name, output_name, config_type, device):
        model = get_test_diff_partitioning(device=device)
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype, device=device))
        output = model({param_name: param})
        loss = output[output_name].sum()

        grads = torch.autograd.grad(loss, param, retain_graph=True, allow_unused=True)[0]
        if grads is not None:
            assert torch.all(grads == 0), f"Expected no gradients for {param_name}->{output_name}"

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_forward_backward_match(self, param_name, output_name, config_type, device):
        model = get_test_diff_partitioning(device=device)
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype, device=device))
        output = model({param_name: param})
        loss = output[output_name].sum()

        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]
        assert grads is not None, f"Gradients for {param_name} should not be None"

        param.grad = None
        loss.backward()
        grad_backward = param.grad

        assert grad_backward is not None, f"Backward gradients for {param_name} should not be None"
        assert torch.all(grad_backward == grads), (
            f"Forward and backward gradients for {param_name} should match"
        )
