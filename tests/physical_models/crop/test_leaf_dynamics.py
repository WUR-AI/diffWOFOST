from unittest.mock import patch
import pytest
import torch
import torch.testing
import yaml
from pcse.base.parameter_providers import ParameterProvider
from pcse.engine import Engine
from pcse.models import Wofost72_PP
from diffwofost.physical_models.crop.leaf_dynamics import WOFOST_Leaf_Dynamics
from tests.physical_models.pcse_test_code import EngineTestHelper
from tests.physical_models.pcse_test_code import WeatherDataProviderTestHelper
from .. import phy_data_folder


def prepare_engine_input(file_path):
    inputs = yaml.safe_load(open(file_path))
    agro_management_inputs = inputs["AgroManagement"]
    cropd = inputs["ModelParameters"]

    weather_data_provider = WeatherDataProviderTestHelper(inputs["WeatherVariables"])
    crop_model_params_provider = ParameterProvider(cropdata=cropd)
    external_states = inputs["ExternalStates"]

    # convert parameters to tensors
    crop_model_params_provider.clear_override()
    for name in ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]:
        value = torch.tensor(crop_model_params_provider[name], dtype=torch.float32)
        crop_model_params_provider.set_override(name, value, check=False)

    # convert external states to tensors
    tensor_external_states = [
        {
            k: v if k == 'DAY' else torch.tensor(v, dtype=torch.float32)
            for k, v in item.items()
        }
        for item in external_states
    ]
    return (
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        tensor_external_states
    )


def get_test_data(file_path):
    inputs = yaml.safe_load(open(file_path))
    return inputs["ModelResults"], inputs["Precision"]


def get_test_diff_leaf_model():
    test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
    (
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        external_states
    ) = prepare_engine_input(test_data_path)
    config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")
    return DiffLeafDynamics(
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        config_path,
        external_states
    )


class DiffLeafDynamics(torch.nn.Module):
    def __init__(
            self,
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            config_path,
            external_states
        ):
        super().__init__()
        self.crop_model_params_provider = crop_model_params_provider
        self.weather_data_provider = weather_data_provider
        self.agro_management_inputs = agro_management_inputs
        self.config_path = config_path
        self.external_states = external_states

    def forward(self, params_dict):
        # pass new value of parameters to the model
        for name, value in params_dict.items():
            self.crop_model_params_provider.set_override(name, value, check=False)

        engine = EngineTestHelper(
            self.crop_model_params_provider,
            self.weather_data_provider,
            self.agro_management_inputs,
            self.config_path,
            self.external_states
        )
        engine.run_till_terminate()
        results = engine.get_output()

        return torch.stack(
            [torch.stack([item['LAI'], item['TWLV']]) for item in results]
        ).unsqueeze(0)  # shape: [1, time_steps, 2]


class TestLeafDynamics:
    def test_leaf_dynamics_with_testengine(self):
        """EngineTestHelper and not Engine because it allows to specify `external_states`."""

        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states
        ) = prepare_engine_input(test_data_path)
        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            config_path,
            external_states
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results, expected_precision = get_test_data(test_data_path)

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                abs(reference[var] - model[var]) < precision
                for var, precision in expected_precision.items()
            )

    def test_leaf_dynamics_with_engine(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            _
        ) = prepare_engine_input(test_data_path)

        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

        # Engine does not allows to specify `external_states`
        with pytest.raises(ValueError):
            Engine(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                config_path
            )

    def test_wofost_pp_with_leaf_dynamics(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_potentialproduction_wofost72_01.yaml"
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            _
        ) = prepare_engine_input(test_data_path)

        with patch(
            'pcse.crop.leaf_dynamics.WOFOST_Leaf_Dynamics',
            WOFOST_Leaf_Dynamics
            ):
            model = Wofost72_PP(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs
            )
            model.run_till_terminate()
            actual_results = model.get_output()

        # get expected results from YAML test data
        expected_results, expected_precision = get_test_data(test_data_path)

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                abs(reference[var] - model[var]) < precision
                for var, precision in expected_precision.items()
            )


class TestDiffLeafDynamicsTDWI:
    def test_gradients_tdwi_lai_leaf_dynamics(self):
        model = get_test_diff_leaf_model()
        tdwi = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        output = model({"TDWI": tdwi})
        lai = output[0, :, 0]
        loss = lai.sum()
        grads = torch.autograd.grad(loss, tdwi, retain_graph=True)[0]  # this is ∂loss/∂tdwi

        assert grads is not None, "Gradients for TDWI should not be None"
        torch.testing.assert_close(
            grads, torch.tensor(0.0013, dtype=torch.float32), rtol=1e-4, atol=1e-4
        )

        tdwi.grad = None  # clear any existing gradient
        loss.backward()
        grad_backward = tdwi.grad

        assert grad_backward is not None, "Backward gradients for TDWI should not be None"
        assert grad_backward == grads, "Forward and backward gradients for TDWI should match"

    def test_gradients_tdwi_twlv_leaf_dynamics(self):
        # prepare model input
        model = get_test_diff_leaf_model()
        tdwi = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        output = model({"TDWI": tdwi})
        twlv = output[0, :, 1]
        loss = twlv.sum()
        grads = torch.autograd.grad(loss, tdwi, retain_graph=True)[0]  # this is ∂loss/∂tdwi

        assert grads is not None, "Gradients for TDWI should not be None"
        torch.testing.assert_close(
            grads, torch.tensor(5.7904, dtype=torch.float32), rtol=1e-4, atol=1e-4
        )

        tdwi.grad = None  # clear any existing gradient
        loss.backward()
        grad_backward = tdwi.grad

        assert grad_backward is not None, "Backward gradients for TDWI should not be None"
        assert grad_backward == grads, "Forward and backward gradients for TDWI should match"


class TestDiffLeafDynamicsSPAN:
    def test_gradients_span_lai_leaf_dynamics(self):
        # prepare model input
        model = get_test_diff_leaf_model()
        span = torch.nn.Parameter(torch.tensor(30, dtype=torch.float32))
        output = model({"SPAN": span})
        lai = output[0, :, 0]
        loss = lai.sum()
        grads = torch.autograd.grad(loss, span, retain_graph=True)[0]  # this is ∂loss/∂span

        assert grads is not None, "Gradients for SPAN should not be None"
        torch.testing.assert_close(
            grads, torch.tensor(2.5047, dtype=torch.float32), rtol=1e-4, atol=1e-4
        )

        span.grad = None  # clear any existing gradient
        loss.backward()
        grad_backward = span.grad

        assert grad_backward is not None, "Backward gradients for TDWI should not be None"
        assert grad_backward == grads, "Forward and backward gradients for TDWI should match"

    def test_gradients_span_twlv_leaf_dynamics(self):
        # prepare model input
        model = get_test_diff_leaf_model()
        span = torch.nn.Parameter(torch.tensor(30, dtype=torch.float32))
        output = model({"SPAN": span})
        twlv = output[0, :, 1]
        loss = twlv.sum()
        grads = torch.autograd.grad(loss, span, retain_graph=True)[0]  # this is ∂loss/∂span

        assert grads is not None, "Gradients for SPAN should not be None"
        torch.testing.assert_close(
            grads, torch.tensor(-0.2426, dtype=torch.float32), rtol=1e-4, atol=1e-4
        )

        span.grad = None  # clear any existing gradient
        loss.backward()
        grad_backward = span.grad

        assert grad_backward is not None, "Backward gradients for TDWI should not be None"
        assert grad_backward == grads, "Forward and backward gradients for TDWI should match"
