import copy
import warnings
from unittest.mock import patch
import pytest
import torch
from pcse.models import Wofost72_PP
from diffwofost.physical_models.config import Configuration
from diffwofost.physical_models.crop.leaf_dynamics import WOFOST_Leaf_Dynamics
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.utils import calculate_numerical_grad
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from .. import phy_data_folder

# Ignore deprecation warnings from pcse.base.simulationobject
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:pcse.base.simulationobject")

leaf_dynamics_config = Configuration(
    CROP=WOFOST_Leaf_Dynamics,
    OUTPUT_VARS=["LAI", "TWLV"],
)


def get_test_diff_leaf_model():
    test_data_url = f"{phy_data_folder}/test_leafdynamics_wofost72_01.yaml"
    test_data = get_test_data(test_data_url)
    crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
    (crop_model_params_provider, weather_data_provider, agro_management_inputs, external_states) = (
        prepare_engine_input(test_data, crop_model_params)
    )
    return DiffLeafDynamics(
        copy.deepcopy(crop_model_params_provider),
        weather_data_provider,
        agro_management_inputs,
        leaf_dynamics_config,
        copy.deepcopy(external_states),
    )


class DiffLeafDynamics(torch.nn.Module):
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
        )
        engine.run_till_terminate()
        results = engine.get_output()

        return {var: torch.stack([item[var] for item in results]) for var in ["LAI", "TWLV"]}


class TestLeafDynamics:
    leafdynamics_data_urls = [
        f"{phy_data_folder}/test_leafdynamics_wofost72_{i:02d}.yaml"
        for i in range(1, 45)  # there are 44 test files
    ]

    wofost72_data_urls = [
        f"{phy_data_folder}/test_potentialproduction_wofost72_{i:02d}.yaml"
        for i in range(1, 45)  # there are 44 test files
    ]

    @pytest.mark.parametrize("test_data_url", leafdynamics_data_urls)
    def test_leaf_dynamics_with_testengine(self, test_data_url):
        """EngineTestHelper and not Engine because it allows to specify `external_states`."""
        # prepare model input
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
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
            leaf_dynamics_config,
            external_states,
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

    @pytest.mark.parametrize(
        "param", ["TDWI", "SPAN", "RGRLAI", "TBASE", "PERDL", "KDIFTB", "SLATB", "TEMP"]
    )
    def test_leaf_dynamics_with_one_parameter_vector(self, param):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_leafdynamics_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI", "KDIFTB", "SLATB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)

        # Setting a vector (with one value) for the selected parameter
        if param == "TEMP":
            # Vectorize weather variable
            for (_, _), wdc in weather_data_provider.store.items():
                wdc.TEMP = torch.ones(10, dtype=torch.float64) * wdc.TEMP
        elif param in ["KDIFTB", "SLATB"]:
            # AfgenTrait parameters need to have shape (N, M)
            repeated = crop_model_params_provider[param].repeat(10, 1)
            crop_model_params_provider.set_override(param, repeated, check=False)
        else:
            repeated = crop_model_params_provider[param].repeat(10)
            crop_model_params_provider.set_override(param, repeated, check=False)

        if param == "TEMP":
            # Expect error due to incompatible shapes
            # (By defaults parameters are not reshaped following weather variables)
            with pytest.raises(ValueError):
                engine = EngineTestHelper(
                    crop_model_params_provider,
                    weather_data_provider,
                    agro_management_inputs,
                    leaf_dynamics_config,
                    external_states,
                )
                engine.run_till_terminate()
                actual_results = engine.get_output()
        else:
            engine = EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                leaf_dynamics_config,
                external_states,
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
            ("TDWI", 0.1),
            ("SPAN", 5),
            ("TBASE", 2.0),
            ("PERDL", 0.01),
            ("RGRLAI", 0.002),
            ("KDIFTB", 0.1),
            ("SLATB", 0.0005),
        ],
    )
    def test_leaf_dynamics_with_different_parameter_values(self, param, delta):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_leafdynamics_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI", "KDIFTB", "SLATB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        # Setting a vector with multiple values for the selected parameter
        test_value = crop_model_params_provider[param]
        # We set the value for which test data are available as the last element
        if param in {"KDIFTB", "SLATB"}:
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
            leaf_dynamics_config,
            external_states,
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

    def test_leaf_dynamics_with_multiple_parameter_vectors(self):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_leafdynamics_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI", "KDIFTB", "SLATB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        # Setting a vector (with one value) for the TDWI and SPAN parameters
        for param in ("TDWI", "SPAN", "RGRLAI", "TBASE", "PERDL", "KDIFTB", "SLATB"):
            if param in ("KDIFTB", "SLATB"):
                # AfgenTrait parameters need to have shape (N, M)
                repeated = crop_model_params_provider[param].repeat(10, 1)
            else:
                repeated = crop_model_params_provider[param].repeat(10)
            crop_model_params_provider.set_override(param, repeated, check=False)

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            leaf_dynamics_config,
            external_states,
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

    def test_leaf_dynamics_with_multiple_parameter_arrays(self):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_leafdynamics_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI", "KDIFTB", "SLATB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)

        # Setting an array with arbitrary shape (and one value)
        for param in ("RGRLAI", "TBASE", "PERDL", "KDIFTB", "SLATB"):
            if param in ("KDIFTB", "SLATB"):
                # AfgenTrait parameters need to have shape (N, M)
                repeated = crop_model_params_provider[param].repeat(30, 5, 1)
            else:
                repeated = crop_model_params_provider[param].broadcast_to((30, 5))
            crop_model_params_provider.set_override(param, repeated, check=False)

        for (_, _), wdc in weather_data_provider.store.items():
            wdc.TEMP = torch.ones((30, 5), dtype=torch.float64) * wdc.TEMP

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            leaf_dynamics_config,
            external_states,
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

    def test_leaf_dynamics_with_incompatible_parameter_vectors(self):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_leafdynamics_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI", "KDIFTB", "SLATB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        # Setting a vector (with one value) for the TDWI and SPAN parameters,
        # but with different lengths
        crop_model_params_provider.set_override(
            "TDWI", crop_model_params_provider["TDWI"].repeat(10), check=False
        )
        crop_model_params_provider.set_override(
            "SPAN", crop_model_params_provider["SPAN"].repeat(5), check=False
        )

        with pytest.raises(AssertionError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                leaf_dynamics_config,
                external_states,
            )

    def test_leaf_dynamics_with_incompatible_weather_parameter_vectors(self):
        # prepare model input
        test_data_url = f"{phy_data_folder}/test_leafdynamics_wofost72_01.yaml"
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI", "KDIFTB", "SLATB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params, meteo_range_checks=False)

        # Setting vectors with incompatible shapes: TDWI and TEMP
        crop_model_params_provider.set_override(
            "TDWI", crop_model_params_provider["TDWI"].repeat(10), check=False
        )
        for (_, _), wdc in weather_data_provider.store.items():
            wdc.TEMP = torch.ones(5, dtype=torch.float64) * wdc.TEMP

        with pytest.raises(ValueError):
            EngineTestHelper(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                leaf_dynamics_config,
                external_states,
            )

    @pytest.mark.parametrize("test_data_url", wofost72_data_urls)
    def test_wofost_pp_with_leaf_dynamics(self, test_data_url):
        # prepare model input
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI", "KDIFTB", "SLATB"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data, crop_model_params)
        )

        # get expected results from YAML test data
        expected_results, expected_precision = test_data["ModelResults"], test_data["Precision"]

        with patch("pcse.crop.wofost72.Leaf_Dynamics", WOFOST_Leaf_Dynamics):
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

    @pytest.mark.parametrize("test_data_url", leafdynamics_data_urls)
    def test_leaf_dynamics_with_sigmoid_approx(self, test_data_url):
        """Test if sigmoid approximation gives same results as leaf dynamics.

        This should be the case since WOFOST_Leaf_Dynamics uses STE method.
        In practice, no approximation is done when not insterested in gradients.
        """
        # prepare model input
        test_data = get_test_data(test_data_url)
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI", "KDIFTB", "SLATB"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data, crop_model_params)

        # Make SPAN a parameter requiring gradients
        crop_model_params_provider["SPAN"].requires_grad = True

        engine = EngineTestHelper(
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            leaf_dynamics_config,
            external_states,
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


class TestDiffLeafDynamicsGradients:
    """Parametrized tests for gradient calculations in leaf dynamics."""

    # Define parameters and outputs
    param_names = ["TDWI", "SPAN", "RGRLAI", "TBASE", "PERDL", "KDIFTB", "SLATB"]
    output_names = ["LAI", "TWLV"]

    # Define parameter configurations (value, dtype)
    param_configs = {
        "single": {
            "TDWI": (0.2, torch.float64),
            "SPAN": (30, torch.float64),
            "RGRLAI": (0.016, torch.float64),
            "TBASE": (3.0, torch.float64),
            "PERDL": (0.03, torch.float64),
            "KDIFTB": ([[0.0, 0.6, 2.0, 0.6]], torch.float64),
            "SLATB": ([[0.0, 0.002, 2.0, 0.002]], torch.float64),
        },
        "tensor": {
            "TDWI": ([0.1, 0.2, 0.3], torch.float64),
            "SPAN": ([25, 30, 35], torch.float64),
            "RGRLAI": ([-10, 0.08, 1], torch.float64),
            "TBASE": ([-5, 0, 10.0], torch.float64),
            "PERDL": ([-10, 0.1, 15], torch.float64),
            "KDIFTB": (
                [[0.0, 0.5, 10.0, 1.0], [0.0, 0.6, 12.0, 1.2], [0.0, 0.4, 8.0, 0.8]],
                torch.float64,
            ),
            "SLATB": (
                [
                    [0.0, 0.002031, 0.5, 0.002031, 2.0, 0.002031],
                    [0.0, 0.0025, 0.6, 0.0025, 2.5, 0.0025],
                    [0.0, 0.0015, 0.4, 0.0015, 1.5, 0.0015],
                ],
                torch.float64,
            ),
        },
    }

    # Define which parameter-output pairs should have gradients
    # Format: {param_name: [list of outputs that should have gradients]}
    gradient_mapping = {
        "TDWI": ["LAI", "TWLV"],
        "SPAN": ["LAI"],
        "RGRLAI": ["LAI"],
        "TBASE": ["LAI"],
        "PERDL": ["TWLV"],
        "KDIFTB": ["LAI"],
        "SLATB": ["LAI"],
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
        model = get_test_diff_leaf_model()
        value, dtype = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=dtype))
        output = model({param_name: param})
        loss = output[output_name].sum()

        grads = torch.autograd.grad(loss, param, retain_graph=True, allow_unused=True)[0]
        if grads is not None:
            assert torch.all((grads == 0) | torch.isnan(grads)), (
                f"Gradient for {param_name} w.r.t. {output_name} should be zero or NaN"
            )

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_forward_backward_match(self, param_name, output_name, config_type):
        """Test that forward and backward gradients match for parameter-output pairs."""
        model = get_test_diff_leaf_model()
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
        assert torch.all(grad_backward == grads), (
            f"Forward and backward gradients for {param_name} should match"
        )

    @pytest.mark.parametrize("param_name,output_name", gradient_params)
    @pytest.mark.parametrize("config_type", ["single", "tensor"])
    def test_gradients_numerical(self, param_name, output_name, config_type):
        """Test that analytical gradients match numerical gradients."""
        value, _ = self.param_configs[config_type][param_name]
        param = torch.nn.Parameter(torch.tensor(value, dtype=torch.float64))

        # we pass `param` and not `param.data` because we want `requires_grad=True`
        # for parameter `SPAN`
        numerical_grad = calculate_numerical_grad(
            get_test_diff_leaf_model, param_name, param, output_name
        )

        model = get_test_diff_leaf_model()
        output = model({param_name: param})
        loss = output[output_name].sum()

        # this is ∂loss/∂param, for comparison with numerical gradient
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]

        # for span, the numerical gradient can't be equal to the pytorch one
        # because we are using STE method
        if (param_name, output_name) not in {("SPAN", "LAI")}:
            # assert_array_almost_equal(numerical_grad, grads.data, decimal=3)
            torch.testing.assert_close(
                numerical_grad,
                grads,
                rtol=1e-3,
                atol=1e-3,
            )

        # Warn if gradient is zero (but this shouldn't happen for gradient_params)
        if torch.all(grads == 0):
            warnings.warn(
                f"Gradient for parameter '{param_name}' with respect to output "
                + f"'{output_name}' is zero: {grads.data}",
                UserWarning,
            )
