import copy
from unittest.mock import patch
import pytest
import torch
from numpy.testing import assert_array_almost_equal
from pcse.engine import Engine
from pcse.models import Wofost72_PP
from diffwofost.physical_models.crop.root_dynamics import WOFOST_Root_Dynamics
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.utils import calculate_numerical_grad
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from diffwofost.physical_models.afgen import Afgen, AfgenTrait
from .. import phy_data_folder


def get_test_diff_root_model():
    test_data_path = phy_data_folder / "test_rootdynamics_wofost72_01.yaml"
    crop_model_params = ["RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU"]
    (crop_model_params_provider, weather_data_provider, agro_management_inputs, external_states) = (
        prepare_engine_input(test_data_path, crop_model_params)
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

        return {var: torch.stack([item[var] for item in results]) for var in ["RD", "TWRT"]}


class TestRootDynamics:
    def test_root_dynamics_with_testengine(self):
        """EngineTestHelper and not Engine because it allows to specify `external_states`."""

        # prepare model input
        test_data_path = phy_data_folder / "test_rootdynamics_wofost72_01.yaml"
        crop_model_params = ["RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data_path, crop_model_params)
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
        crop_model_params = ["RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data_path, crop_model_params)
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
        crop_model_params = ["RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data_path, crop_model_params)
        )

        # get expected results from YAML test data
        expected_results, expected_precision = get_test_data(test_data_path)

        with patch("pcse.crop.wofost72.Root_Dynamics", WOFOST_Root_Dynamics):
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


# Define parameters and outputs
PARAM_NAMES = ["TDWI", "RDI", "RRI", "RDMCR", "RDMSOL", "IAIRDU", "RDRRTB"]
OUTPUT_NAMES = ["RD", "TWRT"]

# Define parameter configurations (value, dtype)
PARAM_DEFAULT_VAL = {
    "TDWI": (0.2, torch.float32),
    "RDI": (0.2, torch.float32),
    "RRI": (0.2, torch.float32),
    "RDMCR": (0.2, torch.float32),
    "RDMSOL": (0.2, torch.float32),
    "IAIRDU": (0.2, torch.float32),
    "RDRRTB": ([0.0, 0.0, 1.5, 0.02], torch.float32),  # Table parameter: [x1, y1, x2, y2]
}

# Define which parameter-output pairs should have gradients
# Format: {param_name: [list of outputs that should have gradients]}
GRADIENT_MAPPING = {
    "TDWI": ["TWRT"],  # e.g. TDWI affects TWRT so I put it here
    "RDI": ["RD"],
    "RRI": ["RD"],
    "RDMCR": ["RD"],
    "RDMSOL": ["RD"],
    "RDRRTB": ["TWRT"],
}

# Generate all combinations
_gradient_params = []
_no_gradient_params = []
for param_name in PARAM_NAMES:
    for output_name in OUTPUT_NAMES:
        if output_name in GRADIENT_MAPPING.get(param_name, []):
            _gradient_params.append((param_name, output_name))
        else:
            _no_gradient_params.append((param_name, output_name))

# Parametrize decorators for gradient tests
gradient_test_params = pytest.mark.parametrize(
    "param_name,output_name",
    _gradient_params,
)
no_gradient_test_params = pytest.mark.parametrize(
    "param_name,output_name",
    _no_gradient_params,
)


class TestDiffRootDynamicsGradients:
    """Parametrized tests for gradient calculations in root dynamics."""

    @no_gradient_test_params
    def test_no_gradients(self, param_name, output_name):
        """Test cases where parameters should not have gradients for specific outputs."""
        model = get_test_diff_root_model()
        value, dtype = PARAM_DEFAULT_VAL[param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype))
        output = model({param_name: param})
        loss = output[output_name].sum()

        assert loss.grad_fn is None

    @gradient_test_params
    def test_gradients_forward_backward_match(self, param_name, output_name):
        """Test that forward and backward gradients match for parameter-output pairs."""
        model = get_test_diff_root_model()
        value, dtype = PARAM_DEFAULT_VAL[param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype))
        output = model({param_name: param})
        loss = output[output_name].sum()

        # this is ∂loss/∂param
        # this is called forward gradient here because it is calculated without backpropagation.
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]

        assert grads is not None, f"Gradients for {param_name} should not be None"

        param.grad = None  # clear any existing gradient
        loss.backward()

        # this is ∂loss/∂param calculated using backpropagation
        grad_backward = param.grad

        assert grad_backward is not None, f"Backward gradients for {param_name} should not be None"
        assert torch.allclose(grad_backward, grads), (
            f"Forward and backward gradients for {param_name} should match"
        )

    @gradient_test_params
    def test_gradients_numerical(self, param_name, output_name):
        """Test that analytical gradients match numerical gradients."""
        value, _ = PARAM_DEFAULT_VAL[param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))
        numerical_grad = calculate_numerical_grad(
            get_test_diff_root_model, param_name, param.data, output_name
        )

        model = get_test_diff_root_model()
        output = model({param_name: param})
        loss = output[output_name].sum()

        # this is ∂loss/∂param, for comparison with numerical gradient
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]

        # in these tests, grads is very small
        # assert_array_almost_equal(numerical_grad, grads.item(), decimal=3)
        assert_array_almost_equal(numerical_grad, grads.detach().numpy(), decimal=3)
