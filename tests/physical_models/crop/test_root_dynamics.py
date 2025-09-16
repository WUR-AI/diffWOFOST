import copy
from unittest.mock import patch
import pytest
import torch
import torch.testing
import yaml
from numpy.testing import assert_almost_equal
from pcse.base.parameter_providers import ParameterProvider
from pcse.engine import Engine
from pcse.models import Wofost72_PP
from diffwofost.physical_models.crop.root_dynamics import WOFOST_Root_Dynamics
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
    for name in ["RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU"]:
        value = torch.tensor(crop_model_params_provider[name], dtype=torch.float32)
        crop_model_params_provider.set_override(name, value, check=False)

    # convert external states to tensors
    tensor_external_states = [
        {k: v if k == "DAY" else torch.tensor(v, dtype=torch.float32) for k, v in item.items()}
        for item in external_states
    ]
    return (
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        tensor_external_states,
    )


def get_test_data(file_path):
    inputs = yaml.safe_load(open(file_path))
    return inputs["ModelResults"], inputs["Precision"]


def get_test_diff_root_model():
    test_data_path = phy_data_folder / "test_rootdynamics_wofost72_01.yaml"
    (crop_model_params_provider, weather_data_provider, agro_management_inputs, external_states) = (
        prepare_engine_input(test_data_path)
    )
    config_path = str(phy_data_folder / "WOFOST_Root_Dynamics.conf")
    return DiffRootDynamics(
        copy.deepcopy(crop_model_params_provider),
        weather_data_provider,
        agro_management_inputs,
        config_path,
        copy.deepcopy(external_states),
    )


class DiffRootDynamics(torch.nn.Module):
    def __init__(
        self,
        crop_model_params_provider,
        weather_data_provider,
        agro_management_inputs,
        config_path,
        external_states,
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
            self.external_states,
        )
        engine.run_till_terminate()
        results = engine.get_output()

        return torch.stack([torch.stack([item["RD"], item["TWRT"]]) for item in results]).unsqueeze(
            0
        )  # shape: [1, time_steps, 2]


def calculate_numerical_grad(param_name, param_value, output_name):
    delta = 1e-6
    p_plus = param_value.item() + delta
    p_minus = param_value.item() - delta

    model = get_test_diff_root_model()
    output = model({param_name: torch.nn.Parameter(torch.tensor(p_plus, dtype=torch.float64))})
    if output_name == "RD":
        loss_plus = output[0, :, 0].sum()
    elif output_name == "TWRT":
        loss_plus = output[0, :, 1].sum()

    model = get_test_diff_root_model()
    output = model({param_name: torch.nn.Parameter(torch.tensor(p_minus, dtype=torch.float64))})
    if output_name == "RD":
        loss_minus = output[0, :, 0].sum()
    elif output_name == "TWRT":
        loss_minus = output[0, :, 1].sum()

    return (loss_plus.item() - loss_minus.item()) / (2 * delta)


class TestRootDynamics:
    def test_root_dynamics_with_testengine(self):
        """EngineTestHelper and not Engine because it allows to specify `external_states`."""

        # prepare model input
        test_data_path = phy_data_folder / "test_rootdynamics_wofost72_01.yaml"
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data_path)
        config_path = str(phy_data_folder / "WOFOST_Root_Dynamics.conf")

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            config_path,
            external_states,
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

    def test_root_dynamics_with_engine(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_rootdynamics_wofost72_01.yaml"
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data_path)
        )

        config_path = str(phy_data_folder / "WOFOST_Root_Dynamics.conf")

        # Engine does not allows to specify `external_states`
        with pytest.raises(KeyError):
            Engine(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                config_path,
            )

    def test_wofost_pp_with_root_dynamics(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_potentialproduction_wofost72_01.yaml"
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data_path)
        )

        with patch("pcse.crop.root_dynamics.WOFOST_Root_Dynamics", WOFOST_Root_Dynamics):
            model = Wofost72_PP(
                crop_model_params_provider, weather_data_provider, agro_management_inputs
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


class TestDiffRootDynamicsTDWI:
    def test_gradients_tdwi_rd_root_dynamics(self):
        model = get_test_diff_root_model()
        tdwi = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        output = model({"TDWI": tdwi})
        rd = output[0, :, 0]
        loss = rd.sum()

        # this is ∂loss/∂tdwi without calling loss.backward().
        # this is called forward gradient here because it is calculated without backpropagation.
        grads = torch.autograd.grad(loss, tdwi, retain_graph=True)[0]

        assert grads is not None, "Gradients for TDWI should not be None"

        tdwi.grad = None  # clear any existing gradient
        loss.backward()

        # this is ∂loss/∂tdwi calculated using backpropagation
        grad_backward = tdwi.grad

        assert grad_backward is not None, "Backward gradients for TDWI should not be None"
        assert grad_backward == grads, "Forward and backward gradients for TDWI should match"

    def test_gradients_tdwi_rd_root_dynamics_numerical(self):
        tdwi = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        numerical_grad = calculate_numerical_grad("TDWI", tdwi, "RD")

        model = get_test_diff_root_model()
        output = model({"TDWI": tdwi})
        rd = output[0, :, 0]
        loss = rd.sum()

        # this is ∂loss/∂tdwi, for comparison with numerical gradient
        grads = torch.autograd.grad(loss, tdwi, retain_graph=True)[0]

        # in this test, grads is very small
        assert_almost_equal(numerical_grad, grads.item(), decimal=3)

    def test_gradients_tdwi_twrt_root_dynamics(self):
        # prepare model input
        model = get_test_diff_root_model()
        tdwi = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        output = model({"TDWI": tdwi})
        twlv = output[0, :, 1]
        loss = twlv.sum()

        # this is ∂loss/∂tdwi
        # this is called forward gradient here because it is calculated without backpropagation.
        grads = torch.autograd.grad(loss, tdwi, retain_graph=True)[0]

        assert grads is not None, "Gradients for TDWI should not be None"

        tdwi.grad = None  # clear any existing gradient
        loss.backward()

        # this is ∂loss/∂tdwi calculated using backpropagation
        grad_backward = tdwi.grad

        assert grad_backward is not None, "Backward gradients for TDWI should not be None"
        assert grad_backward == grads, "Forward and backward gradients for TDWI should match"

    def test_gradients_tdwi_twrt_root_dynamics_numerical(self):
        tdwi = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        numerical_grad = calculate_numerical_grad("TDWI", tdwi, "TWRT")

        model = get_test_diff_root_model()
        output = model({"TDWI": tdwi})
        twrt = output[0, :, 1]
        loss = twrt.sum()

        # this is ∂loss/∂tdwi, for comparison with numerical gradient
        grads = torch.autograd.grad(loss, tdwi, retain_graph=True)[0]

        # in this test, grads is very small
        assert_almost_equal(numerical_grad, grads.item(), decimal=3)
