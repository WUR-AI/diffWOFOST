import copy
import warnings
from unittest.mock import patch
import pytest
import torch
from numpy.testing import assert_array_almost_equal
from pcse.models import Wofost72_PP
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.root_dynamics import WOFOST_Root_Dynamics
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.utils import calculate_numerical_grad
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from .. import phy_data_folder

# Ignore deprecation warnings from pcse.base.simulationobject
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:pcse.base.simulationobject")

root_dynamics_config = Configuration(
    CROP=WOFOST_Root_Dynamics,
    OUTPUT_VARS=["RD", "TWRT"],
)


def get_test_diff_root_model():
    test_data_url = f"{phy_data_folder}/test_rootdynamics_wofost72_01.yaml"
    test_data = get_test_data(test_data_url)
    crop_model_params = ["RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU"]
    (crop_model_params_provider, weather_data_provider, agro_management_inputs, external_states) = (
        prepare_engine_input(test_data, crop_model_params)
    )
    return DiffRootDynamics(
        copy.deepcopy(crop_model_params_provider),
        weather_data_provider,
        agro_management_inputs,
        root_dynamics_config,
        copy.deepcopy(external_states),
    )


class DiffRootDynamics(torch.nn.Module):
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

    def forward(self, params_dict):
        # pass new value of parameters to the model
        for name, value in params_dict.items():
            self.crop_model_params_provider.set_override(name, value, check=False)

        engine = EngineTestHelper(
            self.crop_model_params_provider,
            self.weather_data_provider,
            self.agro_management_inputs,
            self.config,
            self.external_states,
            device="cpu",
        )
        engine.run_till_terminate()
        results = engine.get_output()

        return {var: torch.stack([item[var] for item in results]) for var in ["RD", "TWRT"]}


class TestRootDynamics:
    rootdynamics_data_urls = [
        f"{phy_data_folder}/test_rootdynamics_wofost72_{i:02d}.yaml"
        for i in range(1, 45)  # there are 44 test files
    ]

    wofost72_data_urls = [
        f"{phy_data_folder}/test_potentialproduction_wofost72_{i:02d}.yaml"
        for i in range(1, 45)  # there are 44 test files
    ]

    @pytest.mark.parametrize("test_data_url", rootdynamics_data_urls)
    def test_root_dynamics_with_testengine(self, test_data_url):
        """EngineTestHelper and not Engine because it allows to specify `external_states`."""
        # prepare model input
        test_data = get_test_data(test_data_url)
        crop_model_params = ["RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU", "RDRRTB"]
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
            root_dynamics_config,
            external_states,
            device="cpu",
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                abs(reference[var] - model[var]) < precision
                for var, precision in expected_precision.items()
            )

    @pytest.mark.parametrize("param", ["RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU", "RDRRTB"])
    def test_root_dynamics_with_one_parameter_vector(self, param):
        # prepare model input
        test_data_url = phy_data_folder / "test_rootdynamics_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU", "RDRRTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        # Setting a vector (with one value) for the selected parameter
        # If the parameter is an Afgen table (like RDRRTB), the repeat will create a
        # tensor of Afgen objects
        if param == "RDRRTB":
            repeated = crop_model_params_provider[param].repeat(10, 1)
        else:
            repeated = crop_model_params_provider[param].repeat(10)
        crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            root_dynamics_config,
            external_states,
            device="cpu",
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                all(abs(reference[var] - model[var]) < precision)
                for var, precision in expected_precision.items()
            )

    @pytest.mark.parametrize(
        "param,delta",
        [
            ("RDI", 1.0),
            ("RRI", 0.1),
            ("RDMCR", 10.0),
            ("RDMSOL", 10.0),
            ("TDWI", 0.05),
            ("IAIRDU", 0.05),
            ("RDRRTB", 0.01),
        ],
    )
    def test_root_dynamics_with_different_parameter_values(self, param, delta):
        # prepare model input
        test_data_url = phy_data_folder / "test_rootdynamics_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU", "RDRRTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        # Setting a vector with multiple values for the selected parameter
        test_value = crop_model_params_provider[param]
        # We set the value for which test data are available as the last element
        if param == "RDRRTB":
            # AfgenTrait parameters need to have shape (N, M)
            non_zeros_mask = test_value != 0
            param_vec = torch.stack([test_value + non_zeros_mask * delta, test_value])
        else:
            param_vec = torch.tensor([test_value - delta, test_value + delta, test_value])
        crop_model_params_provider.set_override(param, param_vec, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            root_dynamics_config,
            external_states,
            device="cpu",
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                # The value for which test data are available is the last element
                abs(reference[var] - model[var][-1]) < precision
                for var, precision in expected_precision.items()
            )

    def test_root_dynamics_with_multiple_parameter_vectors(self):
        # prepare model input
        test_data_url = phy_data_folder / "test_rootdynamics_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU", "RDRRTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        # Setting a vector (with one value) for the RDI and RRI parameters
        for param in ("RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU", "RDRRTB"):
            # If the parameter is an Afgen table (like RDRRTB), the repeat will create a
            # tensor of Afgen objects
            if param == "RDRRTB":
                repeated = crop_model_params_provider[param].repeat(10, 1)
            else:
                repeated = crop_model_params_provider[param].repeat(10)
            crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            root_dynamics_config,
            external_states,
            device="cpu",
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        assert len(actual_results) == len(expected_results)

        for reference, model in zip(expected_results, actual_results, strict=False):
            assert reference["DAY"] == model["day"]
            assert all(
                all(abs(reference[var] - model[var]) < precision)
                for var, precision in expected_precision.items()
            )

    def test_root_dynamics_with_multiple_parameter_arrays(self):
        # prepare model input
        test_data_url = phy_data_folder / "test_rootdynamics_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU", "RDRRTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        # Setting an array with arbitrary shape (and one value)
        for param in ("RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU", "RDRRTB"):
            if param == "RDRRTB":
                repeated = crop_model_params_provider[param].repeat((30, 5, 1))
            else:
                repeated = crop_model_params_provider[param].broadcast_to((30, 5))
            crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            root_dynamics_config,
            external_states,
            device="cpu",
        )
        engine.run_till_terminate()
        actual_results = engine.get_output()

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

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

    def test_root_dynamics_with_incompatible_parameter_vectors(self):
        # prepare model input
        test_data_url = phy_data_folder / "test_rootdynamics_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU", "RDRRTB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        # Setting a vector (with one value) for the RDI and RRI parameters,
        # but with different lengths
        crop_model_params_provider.set_override(
            "RDI", crop_model_params_provider["RDI"].repeat(10), check=False
        )
        crop_model_params_provider.set_override(
            "RRI", crop_model_params_provider["RRI"].repeat(5), check=False
        )

        with pytest.raises(AssertionError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                root_dynamics_config,
                external_states,
                device="cpu",
            )

    @pytest.mark.parametrize("test_data_url", wofost72_data_urls)
    def test_wofost_pp_with_root_dynamics(self, test_data_url):
        # prepare model input
        test_data = get_test_data(test_data_url)
        crop_model_params = ["RDI", "RRI", "RDMCR", "RDMSOL", "TDWI", "IAIRDU", "RDRRTB"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data, crop_model_params)
        )

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

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


class TestDiffRootDynamicsGradients:
    """Parametrized tests for gradient calculations in root dynamics."""

    # Define parameters and outputs
    param_names = ["TDWI", "RDI", "RRI", "RDMCR", "RDMSOL", "IAIRDU", "RDRRTB"]
    output_names = ["RD", "TWRT"]

    # Define parameter configurations (value, dtype)
    param_configs = {
        "single": {
            "TDWI": (0.2, torch.float32),
            "RDI": (10.1, torch.float32),
            "RRI": (2.25, torch.float32),
            "RDMCR": (121, torch.float32),
            "RDMSOL": (121, torch.float32),
            "IAIRDU": (0.2, torch.float32),
            "RDRRTB": ([0.0, 0.0, 1.5, 0.02], torch.float32),
        },
        "tensor": {
            "TDWI": ([0.2, 0.3, 0.5], torch.float32),
            "RDI": ([10, 10.1], torch.float32),
            "RRI": ([1.0, 1.5, 2.0, 2.25], torch.float32),
            "RDMCR": ([120, 121], torch.float32),
            "RDMSOL": ([120, 121], torch.float32),
            "IAIRDU": ([0.2, 0.2, 0.3, 0.4], torch.float32),
            "RDRRTB": ([[0.0, 0.0, 1.5, 0.02], [0.0, 0.0, 1.6, 0.03]], torch.float32),
        },
    }

    # Define which parameter-output pairs should have gradients
    # Format: {param_name: [list of outputs that should have gradients]}
    gradient_mapping = {
        "TDWI": ["TWRT"],  # e.g. TDWI affects TWRT so I put it here
        "RDI": ["RD"],
        "RRI": ["RD"],
        "RDMCR": ["RD"],
        "RDMSOL": ["RD"],
        "RDRRTB": ["TWRT"],
    }

    # Generate all combinations
    gradient_params = []
    no_gradient_params = []
    for param_name in param_names:
        for output_name in output_names:
            if output_name in gradient_mapping.get(param_name, []):
                gradient_params.append((param_name, output_name))
            else:
                no_gradient_params.append((param_name, output_name))

    @pytest.mark.parametrize("param_name,output_name", no_gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_no_gradients(self, param_name, output_name, config_type):
        """Test cases where parameters should not have gradients for specific outputs."""
        model = get_test_diff_root_model()
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype))
        output = model({param_name: param})
        loss = output[output_name].sum()

        assert loss.grad_fn is None

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_forward_backward_match(self, param_name, output_name, config_type):
        """Test that forward and backward gradients match for parameter-output pairs."""
        model = get_test_diff_root_model()
        value, dtype = self.param_configs[config_type][param_name]
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

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_numerical(self, param_name, output_name, config_type):
        """Test that analytical gradients match numerical gradients."""
        value, _ = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))
        numerical_grad = calculate_numerical_grad(
            get_test_diff_root_model, param_name, param.data, output_name
        )

        model = get_test_diff_root_model()
        output = model({param_name: param})
        loss = output[output_name].sum()

        # this is ∂loss/∂param, for comparison with numerical gradient
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]

        assert_array_almost_equal(numerical_grad, grads.detach().numpy(), decimal=3)

        # Warn if gradient is zero
        if torch.all(grads == 0):
            warnings.warn(
                f"Gradient for parameter '{param_name}' with respect to output"
                + f"'{output_name}' is zero: {grads.detach().numpy()}",
                UserWarning,
            )
