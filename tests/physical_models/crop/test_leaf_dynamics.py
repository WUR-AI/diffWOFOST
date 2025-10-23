import copy
from unittest.mock import patch
import pytest
import torch
from numpy.testing import assert_array_almost_equal
from pcse.engine import Engine
from pcse.models import Wofost72_PP
from diffwofost.physical_models.crop.leaf_dynamics import WOFOST_Leaf_Dynamics
from diffwofost.physical_models.utils import EngineTestHelper
from diffwofost.physical_models.utils import calculate_numerical_grad
from diffwofost.physical_models.utils import get_test_data
from diffwofost.physical_models.utils import prepare_engine_input
from .. import phy_data_folder


def get_test_diff_leaf_model():
    test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
    crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
    (crop_model_params_provider, weather_data_provider, agro_management_inputs, external_states) = (
        prepare_engine_input(test_data_path, crop_model_params)
    )
    config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")
    return DiffLeafDynamics(
        copy.deepcopy(crop_model_params_provider),
        weather_data_provider,
        agro_management_inputs,
        config_path,
        copy.deepcopy(external_states),
    )


class DiffLeafDynamics(torch.nn.Module):
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

        return {var: torch.stack([item[var] for item in results]) for var in ["LAI", "TWLV"]}


class TestLeafDynamics:
    def test_leaf_dynamics_with_testengine(self):
        """EngineTestHelper and not Engine because it allows to specify `external_states`."""

        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data_path, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

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

    def test_leaf_dynamics_with_engine(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data_path, crop_model_params)
        )

        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

        # Engine does not allows to specify `external_states`
        with pytest.raises(ValueError):
            Engine(
                crop_model_params_provider,
                weather_data_provider,
                agro_management_inputs,
                config_path,
            )

    @pytest.mark.parametrize("param", ["TDWI", "SPAN"])
    def test_leaf_dynamics_with_one_parameter_vector(self, param):
        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data_path, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

        # Setting a vector (with one value) for the selected parameter
        repeated = crop_model_params_provider[param].repeat(10)
        crop_model_params_provider.set_override(param, repeated, check=False)

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
                all(abs(reference[var] - model[var]) < precision)
                for var, precision in expected_precision.items()
            )

    @pytest.mark.parametrize("param,delta", [
        ("TDWI", 0.1),
        ("SPAN", 5),
    ])
    def test_leaf_dynamics_with_different_parameter_values(self, param, delta):
        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data_path, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

        # Setting a vector with multiple values for the selected parameter
        test_value = crop_model_params_provider[param]
        # We set the value for which test data are available as the last element
        param_vec = torch.tensor([test_value - delta, test_value + delta, test_value])
        crop_model_params_provider.set_override(param, param_vec, check=False)

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
                # The value for which test data are available is the last element
                abs(reference[var] - model[var][-1]) < precision
                for var, precision in expected_precision.items()
            )

    def test_leaf_dynamics_with_multiple_parameter_vectors(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data_path, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

        # Setting a vector (with one value) for the TDWI and SPAN parameters
        for param in ("TDWI", "SPAN"):
            repeated = crop_model_params_provider[param].repeat(10)
            crop_model_params_provider.set_override(param, repeated, check=False)

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
                all(abs(reference[var] - model[var]) < precision)
                for var, precision in expected_precision.items()
            )

    def test_leaf_dynamics_with_multiple_parameter_arrays(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data_path, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

        # Setting an array with arbitrary shape (and one value) for the
        # TDWI and SPAN parameters
        for param in ("TDWI", "SPAN"):
            repeated = crop_model_params_provider[param].broadcast_to((30, 5))
            crop_model_params_provider.set_override(param, repeated, check=False)

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
                torch.all(abs(reference[var] - model[var]) < precision)
                for var, precision in expected_precision.items()
            )
            assert all(
                model[var].shape == (30, 5) for var in expected_precision.keys()
            )  # check the output shapes

    def test_leaf_dynamics_with_incompatible_parameter_vectors(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_leafdynamics_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (
            crop_model_params_provider,
            weather_data_provider,
            agro_management_inputs,
            external_states,
        ) = prepare_engine_input(test_data_path, crop_model_params)
        config_path = str(phy_data_folder / "WOFOST_Leaf_Dynamics.conf")

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
                config_path,
                external_states,
            )

    def test_wofost_pp_with_leaf_dynamics(self):
        # prepare model input
        test_data_path = phy_data_folder / "test_potentialproduction_wofost72_01.yaml"
        crop_model_params = ["SPAN", "TDWI", "TBASE", "PERDL", "RGRLAI"]
        (crop_model_params_provider, weather_data_provider, agro_management_inputs, _) = (
            prepare_engine_input(test_data_path, crop_model_params)
        )

        # get expected results from YAML test data
        expected_results, expected_precision = get_test_data(test_data_path)

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


class TestDiffLeafDynamics:
    @pytest.mark.parametrize("param_name,param_value,out_name", [
        ("TDWI", torch.tensor(0.2, dtype=torch.float64), "LAI"),
        ("TDWI", torch.tensor(0.2, dtype=torch.float64), "TWLV"),
        ("TDWI", torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64), "LAI"),
        ("TDWI", torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64), "TWLV"),
        ("SPAN", torch.tensor(30, dtype=torch.float64), "LAI"),
        ("SPAN", torch.tensor(30, dtype=torch.float64), "TWLV"),
        ("SPAN", torch.tensor([25, 30, 35], dtype=torch.float64), "LAI"),
        ("SPAN", torch.tensor([25, 30, 35], dtype=torch.float64), "TWLV"),
    ])
    def test_gradients_leaf_dynamics(self, param_name, param_value, out_name):
        model = get_test_diff_leaf_model()
        param = torch.nn.Parameter(param_value)
        output = model({param_name: param})
        loss = output[out_name].sum()

        # this is ∂loss/∂param without calling loss.backward().
        # this is called forward gradient here because it is calculated without backpropagation.
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]
        assert grads is not None, "Gradients should not be None"

        param.grad = None  # clear any existing gradient
        loss.backward()
        # this is ∂loss/∂param calculated using backpropagation
        grad_backward = param.grad

        assert grad_backward is not None, "Backward gradients should not be None"
        assert torch.all(grad_backward == grads), "Forward and backward gradients should match"

    @pytest.mark.parametrize("param_name,param_value,out_name", [
        ("TDWI", torch.tensor(0.2, dtype=torch.float64), "LAI"),
        ("TDWI", torch.tensor(0.2, dtype=torch.float64), "TWLV"),
        ("TDWI", torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64), "LAI"),
        ("TDWI", torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64), "TWLV"),
        ("SPAN", torch.tensor(30, dtype=torch.float64), "LAI"),
        ("SPAN", torch.tensor(30, dtype=torch.float64), "TWLV"),
        ("SPAN", torch.tensor([25, 30, 35], dtype=torch.float64), "LAI"),
        ("SPAN", torch.tensor([25, 30, 35], dtype=torch.float64), "TWLV"),
    ])
    def test_gradients_leaf_dynamics_numerical(self, param_name, param_value, out_name):
        # first check if the numerical gradient isnot zero i.e. the parameter has an effect
        param = torch.nn.Parameter(param_value)
        numerical_grad = calculate_numerical_grad(
            get_test_diff_leaf_model, param_name, param.data, out_name
        )  # this is Δloss/Δparam

        model = get_test_diff_leaf_model()
        output = model({param_name: param})
        loss = output[out_name].sum()

        # this is ∂loss/∂param, for comparison with numerical gradient
        grads = torch.autograd.grad(loss, param, retain_graph=True)[0]

        assert_array_almost_equal(numerical_grad, grads.data, decimal=3)
