import warnings
from unittest.mock import patch
import pytest
import torch
from numpy.testing import assert_array_almost_equal
from pcse.models import Wofost72_PP
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.wofost72 import Wofost72
from diffwofost.physical_models.soil.classic_waterbalance import WaterbalancePP
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.utils import calculate_numerical_grad
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from .. import phy_data_folder

waterbalance_config = Configuration(
    CROP=Wofost72,
    SOIL=WaterbalancePP,
    # SM is published by WaterbalancePP; EVS is its rate variable.
    OUTPUT_VARS=["SM", "EVS"],
)


def get_test_diff_waterbalance_model(device: str = "cpu"):
    """Return a fresh DiffWaterbalancePP model ready to be called."""
    test_data_url = f"{phy_data_folder}/test_potentialproduction_wofost72_05.yaml"
    test_data = get_test_data(test_data_url)
    crop_model_params = ["SMFCF"]
    (crop_model_params_provider, weather_data_provider, agro_management_inputs, external_states) = (
        prepare_engine_input(test_data, crop_model_params, device=device)
    )
    return DiffWaterbalancePP(
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        waterbalance_config,
        external_states,
    )


class DiffWaterbalancePP(torch.nn.Module):
    def __init__(
        self,
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        config,
        external_states,
    ):
        super().__init__()
        self.crop_model_params_provider = crop_model_params_provider
        self.weather_data_provider = weather_data_provider
        self.agro_management_inputs = agro_management_inputs
        self.config = config
        self.external_states = external_states
        self.engine = EngineTestHelper(config=self.config, external_states=self.external_states)

    def forward(self, params_dict):
        # Pass new parameter values to the model
        for name, value in params_dict.items():
            self.crop_model_params_provider.set_override(name, value, check=False)

        engine = self.engine.setup(
            self.crop_model_params_provider,
            self.weather_data_provider,
            self.agro_management_inputs,
        )
        engine.run_till_terminate()

        return {
            "SM": engine.soil.states.SM,
            "EVS": engine.soil.rates.EVS,
        }


@pytest.mark.usefixtures("fast_mode")
class TestWaterbalancePP:
    waterbalance_data_urls = [
        f"{phy_data_folder}/test_potentialproduction_wofost72_{i:02d}.yaml"
        for i in range(1, 45)  # there are 44 test files
    ]

    @pytest.mark.parametrize("test_data_url", waterbalance_data_urls)
    def test_waterbalance_sm_equals_smfcf(self, test_data_url, device):
        """Soil moisture SM should equal soil moisture factor SMFCF from crop model params."""
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SMFCF"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        smfcf = torch.as_tensor(crop_model_params_provider["SMFCF"], dtype=torch.float64).cpu()

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            waterbalance_config,
            external_states,
        )
        engine.run_till_terminate()

        sm_from_soil = engine.soil.states.SM.cpu()

        assert engine.soil.states.SM.device.type == device, "SM should be on the correct device"
        assert torch.all(torch.isclose(sm_from_soil, smfcf, atol=1e-6)), (
            f"SM={sm_from_soil} should equal SMFCF={smfcf}"
        )

    @pytest.mark.parametrize("test_data_url", waterbalance_data_urls)
    def test_waterbalance_evs_non_negative(self, test_data_url, device):
        """Soil evaporation rate EVS must never be negative."""
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SMFCF"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            waterbalance_config,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        assert len(actual_results) > 0
        for model in actual_results:
            assert model["EVS"].device.type == device, "EVS should be on the correct device"
            assert torch.all(model["EVS"].cpu() >= 0.0), "EVS must be non-negative"

    def test_waterbalance_with_one_parameter_vector(self, device):
        """SMFCF as a 1-D vector → SM is broadcast to the same shape."""
        test_data_url = phy_data_folder / "test_potentialproduction_wofost72_05.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SMFCF"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        repeated = crop_model_params_provider["SMFCF"].repeat(10)
        crop_model_params_provider.set_override("SMFCF", repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            waterbalance_config,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        assert len(actual_results) > 0
        for model in actual_results:
            assert model["EVS"].device.type == device, "EVS should be on the correct device"
            # EVS is computed from weather forcings only (not SMFCF), so it is always
            # scalar regardless of the SMFCF batch size.
            assert torch.all(model["EVS"].cpu() >= 0.0), "EVS must be non-negative"

        sm_from_soil = engine.soil.states.SM.cpu()
        assert sm_from_soil.shape == (10,), "SM from soil should have shape (10,)"
        assert torch.all(torch.isclose(sm_from_soil, repeated.cpu(), atol=1e-6)), (
            "Each SM element should equal the corresponding SMFCF value"
        )

    @pytest.mark.parametrize("delta", [-0.02, 0.02])
    def test_waterbalance_with_different_smfcf_values(self, delta, device):
        """Batch of 3 SMFCF values → SM equals each value independently."""
        test_data_url = phy_data_folder / "test_potentialproduction_wofost72_05.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SMFCF"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        smfcf = crop_model_params_provider["SMFCF"]
        # Place the reference value last (matching the convention in other module tests)
        smfcf_vec = torch.stack([smfcf + delta, smfcf - delta, smfcf])
        crop_model_params_provider.set_override("SMFCF", smfcf_vec, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            waterbalance_config,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        assert len(actual_results) > 0
        for model in actual_results:
            assert model["EVS"].device.type == device, "EVS should be on the correct device"
            # EVS is weather-driven (not SMFCF-dependent), so it is always scalar.
            assert torch.all(model["EVS"].cpu() >= 0.0), "EVS must be non-negative"

        sm_from_soil = engine.soil.states.SM.cpu()
        assert sm_from_soil.shape == (3,), "SM from soil should have shape (3,)"
        assert torch.all(torch.isclose(sm_from_soil, smfcf_vec.cpu(), atol=1e-6)), (
            "SM should equal each SMFCF value in the batch (last element matches reference)"
        )

    def test_waterbalance_with_multiple_parameter_vectors(self, device):
        """SMFCF repeated 10× → EVS and internal SM broadcast correctly."""
        test_data_url = phy_data_folder / "test_potentialproduction_wofost72_05.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SMFCF"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        repeated = crop_model_params_provider["SMFCF"].repeat(10)
        crop_model_params_provider.set_override("SMFCF", repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            waterbalance_config,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        assert len(actual_results) > 0
        for model in actual_results:
            # EVS is weather-driven (not SMFCF-dependent), so it is always scalar.
            assert torch.all(model["EVS"].cpu() >= 0.0), "EVS must be non-negative"

        sm_from_soil = engine.soil.states.SM.cpu()
        assert sm_from_soil.shape == (10,), "SM from soil should have shape (10,)"
        assert torch.all(torch.isclose(sm_from_soil, repeated.cpu(), atol=1e-6))

    def test_waterbalance_with_multiple_parameter_arrays(self, device):
        """2-D SMFCF of shape (30, 5) → SM and EVS outputs carry the same shape."""
        test_data_url = phy_data_folder / "test_potentialproduction_wofost72_05.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SMFCF"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        smfcf_2d = crop_model_params_provider["SMFCF"].broadcast_to((30, 5))
        crop_model_params_provider.set_override("SMFCF", smfcf_2d, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            waterbalance_config,
            external_states,
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        assert len(actual_results) > 0
        for model in actual_results:
            # EVS is weather-driven (not SMFCF-dependent), so it is always scalar.
            assert torch.all(model["EVS"].cpu() >= 0.0), "EVS must be non-negative"

        sm_from_soil = engine.soil.states.SM.cpu()
        assert sm_from_soil.shape == (30, 5), "SM from soil should have shape (30, 5)"
        assert torch.all(torch.isclose(sm_from_soil, smfcf_2d.cpu(), atol=1e-6))

    @pytest.mark.parametrize("test_data_url", waterbalance_data_urls)
    def test_wofost72_pp_with_waterbalance(self, test_data_url):
        """WaterbalancePP plugged into Wofost72_PP reproduces PCSE reference results."""
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SMFCF"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data, crop_model_params)
        )

        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        with patch("pcse.crop.wofost72.Wofost72", Wofost72):
            model = Wofost72_PP(
                crop_model_params_provider, weather_data_provider, agro_management_inputs
            )
            model.run_till_terminate()
            actual_results = model.get_output()

            assert len(actual_results) == len(expected_results)

            for reference, model_out in zip(expected_results, actual_results, strict=False):
                assert reference["DAY"] == model_out["day"]
                assert all(
                    abs(reference[var] - model_out[var]) < precision
                    for var, precision in expected_precision.items()
                )


@pytest.mark.usefixtures("fast_mode")
class TestDiffWaterbalancePPGradients:
    """Gradient tests for WaterbalancePP."""

    param_names = ["SMFCF"]
    output_names = ["SM", "EVS"]

    param_configs = {
        "single": {
            "SMFCF": (0.30, torch.float64),
        },
        "tensor": {
            "SMFCF": ([0.26, 0.30, 0.34], torch.float64),
        },
    }

    gradient_mapping = {
        "SMFCF": ["SM", "EVS"],
    }

    gradient_params = []
    no_gradient_params = []
    no_graph_mapping: dict[str, list[str]] = {}
    for param_name in param_names:
        no_graph_outputs: list[str] = []
        for output_name in output_names:
            if output_name in gradient_mapping.get(param_name, []):
                gradient_params.append((param_name, output_name))
            else:
                no_gradient_params.append((param_name, output_name))
                no_graph_outputs.append(output_name)
        if no_graph_outputs:
            no_graph_mapping[param_name] = no_graph_outputs

    @pytest.mark.parametrize("param_name,output_name", no_gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_no_gradients(self, param_name, output_name, config_type, device):
        """Parameters should *not* propagate gradients to certain outputs."""
        model = get_test_diff_waterbalance_model(device=device)
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype, device=device))
        output = model({param_name: param})
        loss = output[output_name].sum()

        should_have_no_graph = output_name in self.no_graph_mapping.get(param_name, [])

        if not loss.requires_grad:
            assert should_have_no_graph, (
                f"Expected a computation graph for {param_name} → {output_name}, "
                f"but loss.requires_grad is False"
            )
            return

        assert not should_have_no_graph, (
            f"Expected no computation graph for {param_name} → {output_name}, "
            f"but loss.requires_grad is True"
        )

        grads = torch.autograd.grad(loss, param, retain_graph=True, allow_unused=True)[0]
        if grads is not None:
            assert torch.all((grads == 0) | torch.isnan(grads)), (
                f"Gradient for {param_name} w.r.t. {output_name} should be zero or NaN"
            )

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_forward_backward_match(self, param_name, output_name, config_type, device):
        """Forward (torch.autograd.grad) and backward (loss.backward) gradients must agree."""
        model = get_test_diff_waterbalance_model(device=device)
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype, device=device))
        output = model({param_name: param})
        loss = output[output_name].sum()

        # Forward gradient (∂loss/∂param without loss.backward)
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]
        assert grads is not None, f"Gradients for {param_name} should not be None"

        param.grad = None  # clear any existing gradient
        loss.backward()
        grad_backward = param.grad

        assert grad_backward is not None, f"Backward gradient for {param_name} should not be None"
        assert torch.allclose(grad_backward, grads), (
            f"Forward and backward gradients for {param_name} → {output_name} should match"
        )

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_numerical(self, param_name, output_name, config_type, device):
        """Analytical gradients must match finite-difference numerical gradients."""
        value, _ = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64, device=device))

        numerical_grad = calculate_numerical_grad(
            lambda: get_test_diff_waterbalance_model(device=device),
            param_name,
            param.data,
            output_name,
        )

        model = get_test_diff_waterbalance_model(device=device)
        output = model({param_name: param})
        loss = output[output_name].sum()

        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]

        assert_array_almost_equal(
            numerical_grad.detach().cpu().numpy(),
            grads.detach().cpu().numpy(),
            decimal=3,
        )

        if torch.all(grads == 0):
            warnings.warn(
                f"Gradient for parameter '{param_name}' with respect to output "
                f"'{output_name}' is zero: {grads.detach().cpu().numpy()}",
                UserWarning,
            )
